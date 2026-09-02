import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 880
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, w_ptr, out_ptr, n_cols, stride_x, stride_o,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b

    # softmax
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # cast to fp16 then back to f32 to match reference (softmax output in fp16)
    sm16 = sm.to(tl.float16)
    xf = sm16.to(tl.float32)

    # RMS norm
    ms = tl.sum(xf * xf, axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * r).to(tl.float16)

    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = y * w

    tl.store(out_ptr + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, n_cols)
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_kernel[(x2.shape[0],)](
            x2, self.b0, self.rms2_w, out,
            n_cols, x2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=4,
        )
        return out.view_as(x)
