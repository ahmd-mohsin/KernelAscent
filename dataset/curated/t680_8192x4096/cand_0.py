import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 680
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_softmax_rms_kernel(
    X, W1, W3, OUT,
    stride_xm, stride_om,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulate, round to fp16 like PyTorch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p16 = (e / s).to(tl.float16)
    xf = p16.to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(xf * xf, axis=0) / D
    r = 1.0 / tl.sqrt(ms + EPS)
    y16 = (xf * r).to(tl.float16)

    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    y16 = (y16.to(tl.float32) * w1).to(tl.float16)

    # scale (opmath fp32, round to fp16)
    y16 = (y16.to(tl.float32) * SCALE).to(tl.float16)

    # RMSNorm 2
    yf = y16.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / D
    r2 = 1.0 / tl.sqrt(ms2 + EPS)
    z16 = (yf * r2).to(tl.float16)

    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z16.to(tl.float32) * w3).to(tl.float16)

    tl.store(OUT + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()
        Mrows, Dcols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_rms_kernel[(Mrows,)](
            x, self.rms1_w, self.rms3_w, out,
            x.stride(0), out.stride(0),
            D=Dcols,
            SCALE=1.4345,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
