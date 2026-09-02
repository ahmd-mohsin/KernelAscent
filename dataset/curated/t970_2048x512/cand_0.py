import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 970
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _gelu2_rmsnorm_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # gelu #1 (exact erf, float32 compute like CUDA half opmath), cast back to fp16
    xf = x.to(tl.float32)
    g1 = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865475))
    g1h = g1.to(tl.float16)

    # gelu #2
    g1f = g1h.to(tl.float32)
    g2 = 0.5 * g1f * (1.0 + tl.math.erf(g1f * 0.7071067811865475))
    g2h = g2.to(tl.float16)

    # RMSNorm in float32
    xf2 = g2h.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf2 * xf2, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    normed = (xf2 * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    out = normed * w

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _gelu2_rmsnorm_kernel[(m,)](
            h, self.rms3_w, y,
            n, h.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
