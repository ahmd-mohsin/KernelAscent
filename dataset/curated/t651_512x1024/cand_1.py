import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 651
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _rms_relu_softmax_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * rstd).to(tl.float16)

    # weight multiply in fp16 (matches reference dtype semantics)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float16)
    y16 = xn * w

    # relu
    zero = tl.zeros(y16.shape, dtype=tl.float16)
    y16 = tl.maximum(y16, zero)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_relu_softmax_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            n, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
