import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 541
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_rms_bias_scale(X, W, B, Y, D: tl.constexpr, EPS: tl.constexpr,
                          SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # mean of squares over the row
    ms = tl.sum(x * x, axis=0) / D
    rs = 1.0 / tl.sqrt(ms + EPS)

    # match PyTorch op-by-op rounding to fp16 between each elementwise op
    y = (x * rs).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w).to(tl.float16)

    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) + b).to(tl.float16)

    y = (y.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y + row * D + cols, y, mask=mask)


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
            return x * 1.4119

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_rms_bias_scale[(m,)](
            xc, self.rms0_w, self.b1, y,
            D=d, EPS=1e-6, SCALE=1.4119,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
