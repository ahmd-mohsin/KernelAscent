import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 871
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_kernel(
    X, W2, W3, Y,
    stride_x, stride_y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # relu + scale (compute in fp32, round back to fp16 like torch opmath)
    xf = x.to(tl.float32)
    xf = tl.where(xf > 0.0, xf, 0.0)
    xf = xf * 1.1998
    x = xf.to(tl.float16)

    # RMSNorm 1
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * inv).to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    x = xh * w2

    # RMSNorm 2
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * inv).to(tl.float16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0)
    y = xh * w3

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.1998
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2, self.rms2_w, self.rms3_w, y,
            x2.stride(0), y.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
