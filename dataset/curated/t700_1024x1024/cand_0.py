import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 700
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_softmax_rms2_kernel(
    x_ptr, w2_ptr, w3_ptr, out_ptr,
    D, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- softmax (fp32 accumulation, matching PyTorch half softmax) ----
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y16 = (e / s).to(tl.float16)  # round to fp16 like torch.softmax output
    # relu is a no-op on softmax outputs (all >= 0)

    # ---- RMSNorm 1 ----
    yf = y16.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0)
    y16 = (yf * r).to(tl.float16) * w2  # fp16 multiply, matches torch

    # ---- RMSNorm 2 ----
    yf = y16.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0)
    y16 = (yf * r).to(tl.float16) * w3

    tl.store(out_ptr + row * stride_o + offs, y16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            # fallback (CPU or non-fp16): reference path
            x = torch.softmax(x, dim=-1)
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_softmax_rms2_kernel[(n_rows,)](
            x2d, self.rms2_w, self.rms3_w, out,
            d, x2d.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
