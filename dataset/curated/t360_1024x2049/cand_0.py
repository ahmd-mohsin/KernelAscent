import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 360
M, D, DT = 1024, 2049, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b1_ptr, w_ptr, out_ptr,
                  N, stride_x, stride_o,
                  SCALE: tl.constexpr, EPS: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)

    # x = x * scale + b1, done in fp16 to match reference numerics
    xh = (x * SCALE + b1).to(tl.float16)

    xf = xh.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    y = (xf * inv).to(tl.float16)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = y * w

    tl.store(out_ptr + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.0393
            x = x + self.b1
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x

        x = x.contiguous()
        rows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x, self.b1, self.rms2_w, out,
            N, x.stride(0), out.stride(0),
            SCALE=1.0393, EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
