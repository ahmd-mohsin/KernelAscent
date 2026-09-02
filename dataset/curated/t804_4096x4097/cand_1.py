import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 804
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _rms_bias_kernel(X, W, B, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (x * inv).to(Y.dtype.element_ty)
    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    y = xn * w + b
    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        _rms_bias_kernel[(Mrows,)](
            x, self.rms1_w, self.b2, y, N, 1e-6,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return y
