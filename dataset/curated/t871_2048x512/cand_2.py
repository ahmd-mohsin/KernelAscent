import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 871
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_kernel(x_ptr, w2_ptr, w3_ptr, out_ptr,
                  n_cols, stride_row,
                  EPS: tl.constexpr, SCALE: tl.constexpr,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    # relu (fp16), then scalar mul in fp32 (opmath), round to fp16
    x = tl.maximum(x, 0.0)
    x16 = (x.to(tl.float32) * SCALE).to(tl.float16)

    # RMSNorm 1
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + EPS)
    a16 = (xf * r).to(tl.float16)
    w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0)
    x16 = (a16.to(tl.float32) * w2.to(tl.float32)).to(tl.float16)

    # RMSNorm 2
    xf = x16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + EPS)
    a16 = (xf * r).to(tl.float16)
    w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0)
    y = (a16.to(tl.float32) * w3.to(tl.float32)).to(tl.float16)

    tl.store(out_ptr + row * stride_row + cols, y, mask=mask)


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

        x = x.contiguous()
        rows, cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(cols)
        _fused_kernel[(rows,)](
            x, self.rms2_w, self.rms3_w, out,
            cols, x.stride(0),
            EPS=1e-6, SCALE=1.1998,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
