import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 682
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_rms_gelu_rms(X, W0, W1, Y, N, stride_x, stride_y,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 1 (float32 math, cast to fp16, fp16 weight mul)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (x * r).to(tl.float16)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0).to(tl.float16)
    xh = xh * w0

    # GELU (PyTorch computes half gelu in float32 internally)
    xf = xh.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    xh = g.to(tl.float16)

    # scalar muls (opmath float32, round back to fp16 each time)
    xh = (xh.to(tl.float32) * 1.0992).to(tl.float16)
    xh = (xh.to(tl.float32) * 1.3755).to(tl.float16)

    # RMSNorm 2
    xf = xh.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms2 = tl.sum(xf * xf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    yh = (xf * r2).to(tl.float16)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float16)
    y = yh * w1

    tl.store(Y + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_gelu_rms[(rows,)](
            x2, self.rms0_w, self.rms4_w, y,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
