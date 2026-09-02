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
    X, W, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xn * w)  # fp16 multiply, matching reference

    # softmax with fp32 accumulation (matching PyTorch half softmax)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM
        Mrows, N = x.shape
        out = torch.empty_like(x)
        _rms_softmax_kernel[(Mrows,)](
            x, self.rms1_w, out,
            x.stride(0), out.stride(0),
            N=N, BLOCK=512,
            num_warps=4,
        )
        return out
