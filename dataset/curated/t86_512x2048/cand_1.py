import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 86
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_softmax_rms_softmax(
    x_ptr, b_ptr, w_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # x + b0 (in fp16 like the reference, then upcast for softmax math)
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    xb = (x + b).to(tl.float32)

    # softmax #1 (fp32 accumulate, round result to fp16 like reference output)
    xb = tl.where(mask, xb, float("-inf"))
    m1 = tl.max(xb, 0)
    e1 = tl.exp(xb - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, 0)
    p = (e1 / s1).to(tl.float16).to(tl.float32)

    # RMS norm in fp32, cast to fp16, then multiply by fp16 weight
    ms = tl.sum(p * p, 0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)  # fp16
    y16 = (p * r).to(tl.float16) * w                 # fp16 * fp16 -> fp16
    y = y16.to(tl.float32)

    # softmax #2
    y = tl.where(mask, y, float("-inf"))
    m2 = tl.max(y, 0)
    e2 = tl.exp(y - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, 0)
    out = (e2 / s2).to(tl.float16)

    tl.store(out_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x + self.b0
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_rms_softmax[(rows,)](
            x2, self.b0, self.rms2_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
