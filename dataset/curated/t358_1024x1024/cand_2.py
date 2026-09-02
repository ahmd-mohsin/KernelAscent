import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 358
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _rmsnorm_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + eps)
    # match reference: (x * r) rounded to bf16, then multiplied by w (single-rounded fp32 mul)
    xb = (x * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xb * w).to(tl.bfloat16)
    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 + bias (bf16 add, same rounding order as reference)
        h = (x @ self.W0) + self.b1
        # GEMM 2
        y = h @ self.W2
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _rmsnorm_kernel[(rows,)](
            y, self.rms3_w, out, N, 1e-6,
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 2048 else 4,
        )
        return out
