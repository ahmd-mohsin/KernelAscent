import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 695
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_rms_kernel(
    x_ptr, w2_ptr, w4_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # load and scale (match bf16 rounding of x * 1.4612)
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * 1.4612).to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))

    # softmax in fp32, rounded to bf16
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(x * x, 0) / D
    r = tl.rsqrt(ms + 1e-6)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = ((x * r).to(tl.bfloat16).to(tl.float32) * w2).to(tl.bfloat16).to(tl.float32)

    # scale (bf16 rounding)
    x = (x * 1.1354).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    ms2 = tl.sum(x * x, 0) / D
    r2 = tl.rsqrt(ms2 + 1e-6)
    w4 = tl.load(w4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((x * r2).to(tl.bfloat16).to(tl.float32) * w4).to(tl.bfloat16)

    tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback: reference path
            x = x * 1.4612
            x = torch.softmax(x, dim=-1)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = x * 1.1354
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_softmax_rms_kernel[(n_rows,)](
            x2d, self.rms2_w, self.rms4_w, out,
            D=d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
