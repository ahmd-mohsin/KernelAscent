import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 715
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # normalize in fp32, cast to fp16 (matches .to(dtype))
    xn = (xf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    y = xn * w  # fp16 multiply, matches PyTorch half elementwise mul

    # gelu: PyTorch half gelu upcasts to fp32 internally (opmath)
    t = y.to(tl.float32)
    out = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_gelu_kernel[(m,)](
            x, self.rms1_w, y, n,
            x.stride(0), y.stride(0),
            1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
