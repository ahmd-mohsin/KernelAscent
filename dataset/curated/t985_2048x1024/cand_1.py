import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 985
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    X_ptr, B_ptr, Out_ptr,
    n_cols,
    stride_xm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # x * 1.1389  (fp32 compute, round to bf16 like PyTorch elementwise)
    xf = x.to(tl.float32) * 1.1389
    xf = xf.to(tl.bfloat16).to(tl.float32)

    # x + b1
    xf = xf + b.to(tl.float32)
    xf = xf.to(tl.bfloat16).to(tl.float32)

    # x * 1.0458
    xf = xf * 1.0458
    xf = xf.to(tl.bfloat16).to(tl.float32)

    # relu
    xf = tl.maximum(xf, 0.0)

    # gelu (exact, erf-based, fp32 compute then bf16 round)
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * inv_sqrt2))
    xf = xf.to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    num = tl.exp(xf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(m,)](
            x, self.b1, out,
            n,
            x.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
