import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 70
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _gelu2_scale_softmax_kernel(
    X, Y,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (erf-based, fp32 compute, round back to bf16 like PyTorch)
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    g = g * 0.5 * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale (fp32 compute, round to bf16 like PyTorch elementwise mul)
    g = g * 1.1999
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax over the row (fp32 accumulation like PyTorch)
    g = tl.where(mask, g, float("-inf"))
    row_max = tl.max(g, 0)
    e = tl.math.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, 0)
    y = e / denom

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (already optimal on A100 tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N_)
        num_warps = 4 if BLOCK <= 1024 else 8
        _gelu2_scale_softmax_kernel[(M_,)](
            h, out, N_, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
