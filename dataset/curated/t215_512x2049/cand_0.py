import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 215
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_relu_rms_kernel(X, W, Y, N, stride_x, stride_y, eps,
                           BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * inv).to(x.dtype)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_relu_rms_kernel[(Mrows,)](
            h, self.rms2_w, y, N,
            h.stride(0), y.stride(0), 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
