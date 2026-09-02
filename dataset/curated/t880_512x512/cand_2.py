import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 880
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, b_ptr, w_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load in fp16, add bias in fp16 (matches x + self.b0 in fp16)
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    xb = x + b  # fp16 add

    # softmax with float32 accumulation (matches PyTorch half softmax)
    xf = xb.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    soft = (e / s).to(tl.float16)  # softmax output stored as fp16

    # RMSNorm: recompute in float32 from fp16 softmax values
    sf = soft.to(tl.float32)
    ms = tl.sum(sf * sf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    normed = (sf * inv).to(tl.float16)  # cast to fp16 before weight mul

    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    out = normed * w  # fp16 multiply

    tl.store(out_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.b0, self.rms2_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
