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

    # load input (bf16) -> fp32
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.4612  (compute fp32, round to bf16 like PyTorch scalar mul)
    x = (x * 1.4612).to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulate, like PyTorch), result rounded to bf16
    x_m = tl.where(mask, x, float('-inf'))
    mx = tl.max(x_m, axis=0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # first RMSNorm
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    xb = (x * r).to(tl.bfloat16)

    # * rms2_w (fp32 opmath, round to bf16)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xb = (xb.to(tl.float32) * w2).to(tl.bfloat16)

    # x = x * 1.1354
    x = (xb.to(tl.float32) * 1.1354).to(tl.bfloat16).to(tl.float32)

    # second RMSNorm
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    xb = (x * r).to(tl.bfloat16)

    # * rms4_w
    w4 = tl.load(w4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xb.to(tl.float32) * w4).to(tl.bfloat16)

    tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x * 1.4612
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = x * 1.1354
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        x = x.contiguous()
        rows, d = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_softmax_rms_kernel[(n_rows,)](
            x2, self.rms2_w, self.rms4_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(x.shape)
