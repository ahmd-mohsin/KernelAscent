import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 555
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _rms_gelu_relu_kernel(X, W, Y, N, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS norm (mean of squares in fp32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)

    # normalize, cast to fp16, multiply by weight in fp16
    xn = (x * r).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    h = (xn * w).to(tl.float16)

    # exact GELU computed in fp32 (matches PyTorch half opmath), then relu
    hf = h.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    g = g.to(tl.float16)
    out = tl.maximum(g, 0.0)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_gelu_relu_kernel[(Mrows,)](
            x, self.rms1_w, y, N,
            x.stride(0), y.stride(0),
            1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
