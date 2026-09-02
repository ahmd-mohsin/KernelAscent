import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 274
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_gelu_rms_bias(X, W, B, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU in fp32 (matches PyTorch opmath for fp16), then cast to fp16
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # RMS norm in fp32 over fp16 values
    gf = g16.to(tl.float32)
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    n16 = (gf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    out = n16 * w + b
    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_rms_bias[(m,)](
            h, self.rms2_w, self.b3, y, n, 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return y
