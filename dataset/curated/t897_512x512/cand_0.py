import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 897
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, w2_ptr, w3_ptr, out_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load row, upcast to fp32
    x = tl.load(x_ptr + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))

    # softmax (fp32 accumulation, as PyTorch does for bf16)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # cast down to bf16 (softmax output dtype), then up for rmsnorm
    xb = sm.to(tl.bfloat16)
    xf = xb.to(tl.float32)

    # rmsnorm 1
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0)
    xb = (xf * inv).to(tl.bfloat16) * w2

    # rmsnorm 2
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0)
    y = (xf * inv).to(tl.bfloat16) * w3

    tl.store(out_ptr + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(rows,)](
            x, self.rms2_w, self.rms3_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
