import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 739
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_rms_softmax_kernel(
    X, W, OUT,
    N, stride_xm, stride_om,
    scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32) * scale

    # RMS norm (mean of squares over the row)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # normalize, cast to fp16, multiply by fp16 weight (matches reference dtype math)
    xn = (xf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w  # fp16 multiply

    # softmax in fp32 (matches PyTorch half softmax which accumulates in fp32)
    s = y.to(tl.float32)
    s = tl.where(mask, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N, h.stride(0), out.stride(0),
            1.4683, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
