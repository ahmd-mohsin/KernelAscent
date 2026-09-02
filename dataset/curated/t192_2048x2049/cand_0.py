import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 192
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.1857  (bf16 rounding as in eager op)
    t = (x * 1.1857).to(tl.bfloat16).to(tl.float32)

    # gelu (exact, erf-based), computed in fp32 opmath then rounded to bf16
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # * 1.3931 with bf16 rounding
    g = (g * 1.3931).to(tl.bfloat16).to(tl.float32)

    # relu (no-op after previous relu with positive scale, kept for exactness)
    g = tl.maximum(g, 0.0)

    # softmax in fp32 (matches PyTorch bf16 softmax accumulation)
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100 for bf16)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_act_softmax_kernel[(Mrows,)](
            h, out,
            N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
