import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 129
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _relu_softmax_rms_kernel(
    X, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu in bf16 (matches torch.relu on bf16), then softmax in fp32
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    soft = e / s

    # cast to bf16 (softmax output dtype), then back to fp32 for RMS norm
    soft_bf = soft.to(tl.bfloat16)
    sf = soft_bf.to(tl.float32)

    ms = tl.sum(sf * sf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)

    normed = (sf * r).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = normed * w

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _relu_softmax_rms_kernel[(Mrows,)](
            h, self.rms3_w, y,
            h.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
