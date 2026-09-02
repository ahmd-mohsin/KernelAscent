import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 630
M, D, DT = 4096, 1024, torch.float16

INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_gelu_softmax_gelu_gelu(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row * N + offs
    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf) in fp32, then round to fp16 to match reference
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax (fp32 math, fp16 result to match reference)
    g_masked = tl.where(mask, g, float('-inf'))
    m = tl.max(g_masked, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # GELU
    g2 = 0.5 * sm * (1.0 + tl.math.erf(sm * 0.7071067811865476))
    g2 = g2.to(tl.float16).to(tl.float32)

    # GELU
    g3 = 0.5 * g2 * (1.0 + tl.math.erf(g2 * 0.7071067811865476))

    tl.store(Y + row * N + offs, g3.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        M_, N_ = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_gelu_softmax_gelu_gelu[(M_,)](
            y, out, N_, BLOCK=BLOCK, num_warps=num_warps
        )
        return out
