import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 535
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _rms_gelu2_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (fp32 accumulate), round to bf16, then weight multiply
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    h = (xn.to(tl.float32) * w).to(tl.bfloat16)

    # GELU (exact, erf), fp32 math, round to bf16
    hf = h.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = (0.5 * hf * (1.0 + tl.math.erf(hf * INV_SQRT2))).to(tl.bfloat16)

    # second GELU
    gf = g1.to(tl.float32)
    g2 = (0.5 * gf * (1.0 + tl.math.erf(gf * INV_SQRT2))).to(tl.bfloat16)

    tl.store(Y + row * N + cols, g2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_gelu2_kernel[(Mrows,)](
            x, self.rms1_w, y, N, 1e-6,
            BLOCK=BLOCK, num_warps=8,
        )
        return y
