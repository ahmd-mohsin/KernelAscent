import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 829
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _rms_softmax_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (mean of squares in fp32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)

    # cast normalized value to fp16, then multiply by weight in fp16 (matches reference)
    xn = (xf * r).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y16 = xn * w

    # softmax computed in fp32 (matches PyTorch half softmax semantics)
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_softmax_kernel[(Mrows,)](
            x, self.rms1_w, y,
            N, x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
