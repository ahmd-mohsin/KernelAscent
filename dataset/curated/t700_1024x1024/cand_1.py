import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 700
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_softmax_relu_rms2_kernel(
    X, W2, W3, Out,
    D, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- softmax (fp32 accumulate, fp16 output like PyTorch CUDA softmax) ----
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # ---- relu (identity after softmax, kept for exactness) ----
    p = tl.maximum(p, 0.0)

    # ---- RMSNorm 1 ----
    xf = p.to(tl.float32)
    ms1 = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    y16 = (xf * tl.math.rsqrt(ms1 + 1e-6)).to(tl.float16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    y16 = y16 * w2  # fp16 multiply, matching PyTorch half*half

    # ---- RMSNorm 2 ----
    yf = y16.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / D
    z16 = (yf * tl.math.rsqrt(ms2 + 1e-6)).to(tl.float16)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    out = z16 * w3

    tl.store(Out + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xf0 = torch.softmax(x, dim=-1)
            xf0 = torch.relu(xf0)
            _xf = xf0.float()
            xf0 = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xf0.dtype) * self.rms2_w
            _xf = xf0.float()
            xf0 = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xf0.dtype) * self.rms3_w
            return xf0

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else (8 if BLOCK <= 4096 else 16)

        _fused_softmax_relu_rms2_kernel[(n_rows,)](
            x2, self.rms2_w, self.rms3_w, out,
            d, x2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
