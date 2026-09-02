import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 720
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_rms_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in float32, then round to bf16 (matches reference .to(x.dtype))
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + EPS)
    xn = (xf * inv).to(tl.bfloat16)

    # multiply by weight: bf16*bf16 computed in fp32 (opmath), rounded to bf16
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax computed in fp32, output rounded to bf16
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # relu is no-op on softmax output; scale in fp32, round to bf16
    out = (tl.maximum(sm.to(tl.float32), 0.0) * SCALE).to(tl.bfloat16)
    tl.store(Out_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_softmax_kernel[(Mrows,)](
            x, self.rms1_w, out,
            x.stride(0),
            N=N, EPS=1e-6, SCALE=1.1833, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
