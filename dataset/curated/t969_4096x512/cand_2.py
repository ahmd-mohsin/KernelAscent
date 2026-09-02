import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 969
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _rms_softmax_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / D
    rs = 1.0 / tl.sqrt(ms + 1e-6)

    # cast normalized value to bf16, multiply by bf16 weight (matches reference)
    xn = (xf * rs).to(tl.bfloat16)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.bfloat16)
    t = (xn * w).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    t = tl.where(mask, t, float('-inf'))
    tmax = tl.max(t, axis=0)
    e = tl.exp(t - tmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _rms_softmax_kernel[(m,)](
            x, self.rms0_w, y,
            x.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
