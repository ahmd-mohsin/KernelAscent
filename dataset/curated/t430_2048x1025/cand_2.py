import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 430
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _fused_gelu_softmax_gelu_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based), computed in fp32 then rounded to fp16 (matches PyTorch half gelu)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax (fp32 accumulation, fp16 output — matches torch.softmax on half)
    g_masked = tl.where(mask, g, float('-inf'))
    m = tl.max(g_masked, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # GELU again
    g2 = 0.5 * p * (1.0 + tl.math.erf(p * INV_SQRT2))
    g2 = g2.to(tl.float16).to(tl.float32)

    # scale by 1.1055 (opmath fp32, round to fp16)
    g2 = (g2 * 1.1055).to(tl.float16)

    # RMSNorm in fp32
    xf = g2.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    r = (xf * inv).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float16)
    y = r * w

    tl.store(Y_ptr + row * stride_row + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores, fp32 accumulate)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_softmax_gelu_rms_kernel[(Mrows,)](
            h, self.rms5_w, out,
            N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
