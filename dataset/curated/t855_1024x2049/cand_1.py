import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 855
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    # normalize in fp32, round to fp16 (matches .to(x.dtype))
    xn = (xf * rstd).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    # fp16 multiply, rounds to fp16 (matches fp16 * fp16)
    y16 = xn * w
    # gelu computed in fp32 (matches PyTorch CUDA half gelu opmath)
    yf = y16.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = yf * 0.5 * (1.0 + tl.math.erf(yf * INV_SQRT2))
    tl.store(Y + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        _rms_gelu_kernel[(Mrows,)](
            x, self.rms1_w, y, N, 1e-6,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return y
