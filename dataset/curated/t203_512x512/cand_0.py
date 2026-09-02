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
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # cast to fp16 (matches .to(x.dtype)), then fp32 opmath mult by weight, round to fp16
    n16 = (xf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    t16 = (n16.to(tl.float32) * w).to(tl.float16)
    # scalar mult in fp32 opmath, round to fp16
    out = (t16.to(tl.float32) * scale).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


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

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _rmsnorm_scale_kernel[(Mrows,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            N, 1e-6, 1.3616,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
