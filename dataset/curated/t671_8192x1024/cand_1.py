import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 671
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_bias_softmax_gelu2_rms_kernel(
    X_ptr, B_ptr, W_ptr, Y_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output (bf16) and bias (bf16)
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x + b1  (fp32 opmath, rounded to bf16 like PyTorch)
    z = (x + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matching PyTorch's internal accumulation), output bf16
    z = tl.where(mask, z, float('-inf'))
    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # gelu (erf, fp32 opmath) -> bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = p * 0.5 * (1.0 + tl.math.erf(p * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # second gelu
    g = g * 0.5 * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # rmsnorm: fp32 mean of squares, rsqrt, scale, cast bf16, then * weight
    g_masked = tl.where(mask, g, 0.0)
    ms = tl.sum(g_masked * g_masked, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    normed = (g * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (normed * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (already optimal on A100 tensor cores)
        h = x @ self.W0
        h = h.contiguous()

        rows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_bias_softmax_gelu2_rms_kernel[(rows,)](
            h, self.b1, self.rms5_w, y,
            N=N, BLOCK=BLOCK,
            num_warps=16,
        )
        return y
