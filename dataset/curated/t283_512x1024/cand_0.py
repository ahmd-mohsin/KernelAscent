import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 283
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, B, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then rounded to bf16 (matches PyTorch bf16 gelu)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ReLU
    r = tl.maximum(g, 0.0)
    r = tl.where(mask, r, float('-inf'))

    # softmax in fp32, output rounded to bf16
    m = tl.max(r, axis=0)
    e = tl.exp(r - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    # add bias (fp32 compute, single rounding to bf16)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    o = (sm + b).to(tl.bfloat16).to(tl.float32)

    # scale
    o = (o * 1.2457).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, o, mask=mask)


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
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.view(-1, n_cols)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(n_rows,)](
            x2, self.b3, y, n_cols,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
