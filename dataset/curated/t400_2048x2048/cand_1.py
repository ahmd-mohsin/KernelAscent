import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, W0, W2, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- Load input row (fp16) and weights ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)

    # ---- RMSNorm 0 (compute in fp32, cast to fp16, mul by fp16 weight) ----
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.float16)
    x1 = xn * w0  # fp16 * fp16 -> fp16

    # ---- Softmax (fp32 accumulation, matching PyTorch half softmax) ----
    x1f = tl.where(mask, x1.to(tl.float32), float('-inf'))
    m = tl.max(x1f, axis=0)
    e = tl.exp(x1f - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # ---- RMSNorm 2 ----
    sf = sm.to(tl.float32)
    ms2 = tl.sum(sf * sf, axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    yn = (sf * inv2).to(tl.float16)
    y = yn * w2  # fp16

    # ---- Two scalar multiplies (opmath fp32, cast back to fp16 each time) ----
    y = (y.to(tl.float32) * 1.3308).to(tl.float16)
    y = (y.to(tl.float32) * 1.3952).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.dim() == 2
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.rms0_w, self.rms2_w, out,
            x.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
