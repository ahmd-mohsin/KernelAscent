import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 260
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_relu_bias_rmsnorm(X, B, W, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_
    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    w = tl.load(W + cols, mask=mask, other=0.0)

    # relu + bias in bf16 (matches reference), then cast to fp32 for rmsnorm
    x = tl.maximum(x, 0.0)
    x = x + b
    xf = x.to(tl.float32)

    mean_sq = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(mean_sq + 1e-6)
    # normalize in fp32, cast to bf16, then multiply by weight (matches reference order)
    xn = (xf * inv).to(x.dtype)
    y = xn * w
    tl.store(Y + row * D_ + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x + self.b1
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_relu_bias_rmsnorm[(m,)](
            x, self.b1, self.rms2_w, y, d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
