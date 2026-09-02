import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 794
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b_ptr, out_ptr,
    n_cols,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0   (bf16 op -> round to bf16)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # x = x * 1.4147  (bf16 op -> round to bf16)
    x = (x * 1.4147).to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulate, bf16 output)
    xm = tl.where(mask, x, float("-inf"))
    m1 = tl.max(xm, axis=0)
    e1 = tl.exp(xm - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    x = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # gelu (exact, erf), bf16 output
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = g.to(tl.bfloat16).to(tl.float32)

    # softmax again
    xm = tl.where(mask, x, float("-inf"))
    m2 = tl.max(xm, axis=0)
    e2 = tl.exp(xm - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y = e2 / s2

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x + self.b0
            y = y * 1.4147
            y = torch.softmax(y, dim=-1)
            y = F.gelu(y)
            return torch.softmax(y, dim=-1)

        x = x.contiguous()
        n_rows, n_cols = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, n_cols)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(x2.shape[0],)](
            x2, self.b0, out,
            n_cols, x2.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view_as(x)
