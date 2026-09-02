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
    X_ptr, W2_ptr, W4_ptr, Y_ptr,
    n_cols,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)  # fp16

    # scale + relu (kept in fp16 like reference)
    x = (x * 1.0531).to(tl.float16)
    x = tl.maximum(x, 0.0)

    # RMSNorm #1 (fp32 math, cast to fp16, multiply by fp16 weight)
    xf = x.to(tl.float32)
    sq = tl.where(mask, xf * xf, 0.0)
    ms = tl.sum(sq, axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0)  # fp16
    x = ((xf * r).to(tl.float16) * w2).to(tl.float16)

    # Softmax (fp32 accumulation, as PyTorch does for half inputs)
    xf2 = x.to(tl.float32)
    xf2 = tl.where(mask, xf2, float('-inf'))
    mmax = tl.max(xf2, axis=0)
    e = tl.exp(xf2 - mmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # RMSNorm #2
    pf = p.to(tl.float32)
    sq2 = tl.where(mask, pf * pf, 0.0)
    ms2 = tl.sum(sq2, axis=0) / n_cols
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w4 = tl.load(W4_ptr + offs, mask=mask, other=0.0)  # fp16
    out = ((pf * r2).to(tl.float16) * w4).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            # reference fallback
            x = x * 1.0531
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_row_kernel[(n_rows,)](
            x2d, self.rms2_w, self.rms4_w, y,
            n_cols,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
