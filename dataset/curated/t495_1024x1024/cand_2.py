import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 495
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_relu_rms_softmax(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    # rmsnorm (compute in fp32, cast back to fp16, multiply by weight in fp16)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    xn = (xf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    h = xn * w  # fp16 multiply, matches reference

    # softmax (torch computes in fp32 internally for reductions; emulate carefully)
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, float('-inf'))
    m_ = tl.max(hf, axis=0)
    e = tl.exp(hf - m_)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_relu_rms_softmax[(Mrows,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
