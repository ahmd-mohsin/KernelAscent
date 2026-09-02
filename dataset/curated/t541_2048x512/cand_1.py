import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 541
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _rms_bias_scale_kernel(
    X, W, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    # replicate: (xf * rsqrt).to(fp16) * w  (fp16 math), + b (fp16), * scale (fp16)
    norm = (xf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float16)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float16)

    y = norm * w
    y = y + b
    y = y * SCALE.to(tl.float16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x + self.b1
            x = x * 1.4119
            return x

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8

        _rms_bias_scale_kernel[(m,)](
            x2, self.rms0_w, self.b1, y,
            x2.stride(0), y.stride(0),
            N=n, EPS=1e-6, SCALE=1.4119,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
