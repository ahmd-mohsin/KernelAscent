import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 274
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _gelu_rms_kernel(X, W, B, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU in fp32, then round to fp16 (matches F.gelu on half)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)

    # RMS norm in fp32 over the fp16-rounded gelu output
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)

    y16 = (gf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    t16 = (y16.to(tl.float32) * w).to(tl.float16)

    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (t16.to(tl.float32) + b).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _gelu_rms_kernel[(m,)](
            h, self.rms2_w, self.b3, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
