import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 869
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_row_kernel(
    x_ptr, w2_ptr, w4_ptr, out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load row (fp16) ----
    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0)

    # ---- x = x * 1.0531 (PyTorch computes half ops in float opmath, rounds to half) ----
    xf = x.to(tl.float32) * 1.0531
    x = xf.to(tl.float16)

    # ---- relu ----
    x = tl.maximum(x, 0.0)

    # ---- RMSNorm #1 (compute in fp32, cast to fp16, mul by weight) ----
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0)
    x = (xf * r).to(tl.float16) * w2  # fp16 result

    # ---- softmax over row (fp32 accumulation, fp16 output) ----
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16)

    # ---- RMSNorm #2 ----
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    w4 = tl.load(w4_ptr + offs, mask=mask, other=0.0)
    y = (xf * r).to(tl.float16) * w4

    tl.store(out_ptr + row * stride_row + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x * 1.0531
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        x = x.contiguous()
        rows, N = x.shape[0], x.shape[-1]
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_row_kernel[(rows,)](
            x, self.rms2_w, self.rms4_w, out,
            N, x.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
