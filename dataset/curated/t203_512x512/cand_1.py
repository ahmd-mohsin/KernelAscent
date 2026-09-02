import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 203
M, D, DT = 512, 512, torch.float16


@triton.jit
def _rmsnorm_scale_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    # normalize in fp32, cast to fp16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float16)

    # fp16 multiply by weight, then fp16 multiply by scalar (matches PyTorch order/precision)
    y = xn * w
    y = y * scale.to(tl.float16)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            return x * 1.3616

        x2d = x.contiguous().view(-1, x.shape[-1])
        m, n = x2d.shape
        y = torch.empty_like(x2d)
        BLOCK_N = triton.next_power_of_2(n)
        _rmsnorm_scale_kernel[(m,)](
            x2d, self.rms0_w, y,
            x2d.stride(0), y.stride(0),
            n, 1e-6, 1.3616,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y.view_as(x)
