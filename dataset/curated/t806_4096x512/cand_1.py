import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 806
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b0_ptr, b3_ptr, out_ptr, n_cols, stride_row,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)
    b3 = tl.load(b3_ptr + cols, mask=mask, other=0.0)

    # x + b0 (bf16 add, matching reference rounding)
    t = x + b0  # bf16
    tf = t.to(tl.float32)

    # exact GELU: computed in fp32, rounded to bf16 (matches PyTorch bf16 gelu)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * tf * (1.0 + tl.math.erf(tf * INV_SQRT2))
    g = g.to(tl.bfloat16)

    # relu (exact in any precision)
    g = tl.maximum(g, 0.0)

    # + b3 in bf16
    s = (g + b3).to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax accumulation)
    s = tl.where(mask, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = (e / denom).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            t = x + self.b0
            t = F.gelu(t)
            t = torch.relu(t)
            t = t + self.b3
            return torch.softmax(t, dim=-1)

        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(n_rows,)](
            x, self.b0, self.b3, out,
            n_cols, x.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
