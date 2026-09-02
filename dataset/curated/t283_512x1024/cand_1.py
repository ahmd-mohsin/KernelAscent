import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 283
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, out_ptr, n_cols, stride_row, SCALE: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact (erf-based) GELU in fp32, rounded to bf16 (matches F.gelu on bf16)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # ReLU (exact, no rounding effect)
    r = tl.maximum(g, tl.zeros_like(g))

    # Softmax: fp32 compute, rounded to bf16 (matches torch softmax on bf16)
    rf = r.to(tl.float32)
    rf = tl.where(mask, rf, float('-inf'))
    m = tl.max(rf, axis=0)
    e = tl.exp(rf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # add bias (fp32 opmath, round to bf16)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)
    t = (sm.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # scale (fp32 opmath, round to bf16)
    o = (t.to(tl.float32) * SCALE).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + cols, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = torch.relu(y)
            y = torch.softmax(y, dim=-1)
            y = y + self.b3
            return y * 1.2457

        x = x.contiguous()
        rows, cols = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        orig_shape = x.shape
        x2 = x.view(-1, cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(cols)
        _fused_kernel[(n_rows,)](
            x2, self.b3, out, cols, x2.stride(0), 1.2457,
            BLOCK=BLOCK, num_warps=8 if BLOCK >= 1024 else 4,
        )
        return out.view(orig_shape)
