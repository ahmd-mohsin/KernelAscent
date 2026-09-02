import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 904
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_epilogue_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * stride_x + cols).to(tl.float32)

    # GELU (erf-based, computed in fp32, rounded to fp16 like PyTorch half kernels)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16)

    # ReLU (fp16)
    r = tl.where(g > 0, g, g * 0)

    # RMSNorm in fp32, cast to fp16, then scale by weight (fp16 opmath in fp32)
    xf = r.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv_rms = tl.math.rsqrt(ms + 1e-6)
    y = (xf * inv_rms).to(tl.float16)

    w = tl.load(W_ptr + cols).to(tl.float32)
    y = (y.to(tl.float32) * w).to(tl.float16)

    # Second GELU
    yf = y.to(tl.float32)
    g2 = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    g2 = g2.to(tl.float16)

    # Softmax (fp32 accumulation, fp16 output)
    z = g2.to(tl.float32)
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + cols, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Heavy matmul via cuBLAS tensor cores
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        _fused_epilogue_kernel[(Mrows,)](
            h, self.rms3_w, out,
            h.stride(0), out.stride(0),
            N=N,
            BLOCK=N,
            num_warps=4,
        )
        return out
