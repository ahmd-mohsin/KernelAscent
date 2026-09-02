import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 920
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_gelu_bias_scale_rms(
    X, B, W, Y,
    stride_xm, stride_ym,
    N, SCALE, EPS,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x16 = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b16 = tl.load(B + cols, mask=mask, other=0.0)
    w16 = tl.load(W + cols, mask=mask, other=0.0)

    # GELU (exact, erf) computed in fp32, rounded back to fp16 (matches F.gelu on half)
    xf = x16.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # add bias in fp16 (half + half)
    a16 = g16 + b16

    # scalar multiply: computed in fp32 (opmath), rounded to fp16
    s16 = (a16.to(tl.float32) * SCALE).to(tl.float16)

    # RMSNorm in fp32
    sf = s16.to(tl.float32)
    sq = tl.where(mask, sf * sf, 0.0)
    mean = tl.sum(sq, axis=0) / N
    r = 1.0 / tl.sqrt(mean + EPS)
    y16 = (sf * r).to(tl.float16)

    # elementwise multiply with weight in fp16
    out = y16 * w16
    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS TF32/FP16 tensor-core matmul
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_gelu_bias_scale_rms[(Mrows,)](
            h, self.b2, self.rms4_w, y,
            h.stride(0), y.stride(0),
            N, 1.3289, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
