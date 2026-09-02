import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 445
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _rms_relu_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    xn = (x * r).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x @ self.W1
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_relu_kernel[(Mrows,)](x, self.rms2_w, y, N, 1e-6,
                                   BLOCK=BLOCK, num_warps=8)
        return y
