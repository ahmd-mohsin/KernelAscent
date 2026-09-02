import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 935
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_gelu_rms_gelu_gelu(X, W, Y, N, stride_x, stride_y, eps,
                              BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu #1 (exact, computed in fp32, rounded to fp16 like PyTorch)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g1 = g1.to(tl.float16).to(tl.float32)

    # RMSNorm over the row (fp32 accumulation)
    ms = tl.sum(tl.where(mask, g1 * g1, 0.0), axis=0) / N
    rs = tl.math.rsqrt(ms + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g1 * rs).to(tl.float16).to(tl.float32) * w
    y = y.to(tl.float16).to(tl.float32)

    # gelu #2
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.float16).to(tl.float32)

    # gelu #3
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 4096, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_rms_gelu_gelu[(Mrows,)](
            h, self.rms2_w, out, N,
            h.stride(0), out.stride(0), 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return out
