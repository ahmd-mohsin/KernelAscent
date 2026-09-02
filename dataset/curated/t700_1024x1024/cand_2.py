import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 700
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_kernel(x_ptr, w2_ptr, w3_ptr, out_ptr, N, stride_row,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask,
                other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch half softmax)
    xmax = tl.max(x, axis=0)
    e = tl.exp(x - xmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to fp16 (softmax output dtype), relu is no-op but keep semantics
    p16 = p.to(tl.float16)
    p16 = tl.maximum(p16, 0.0)

    # RMSNorm 1
    xf = p16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    h16 = (xf * inv).to(tl.float16)
    w2 = tl.load(w2_ptr + cols, mask=mask, other=0.0)
    h16 = h16 * w2

    # RMSNorm 2
    xf2 = h16.to(tl.float32)
    ms2 = tl.sum(xf2 * xf2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    o16 = (xf2 * inv2).to(tl.float16)
    w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0)
    o16 = o16 * w3

    tl.store(out_ptr + row * stride_row + cols, o16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(Mrows,)](
            x, self.rms2_w, self.rms3_w, out, N, x.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
