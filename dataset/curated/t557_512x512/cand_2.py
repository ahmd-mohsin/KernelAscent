import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 557
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b1_ptr, w_ptr, out_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)

    # relu(x) + b1, in bf16 like reference
    x = tl.maximum(x, 0.0)
    x = (x + b1).to(tl.bfloat16)

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * inv).to(tl.bfloat16)

    # * rms2_w (bf16 mul), relu
    y = (y * w).to(tl.bfloat16)
    y = tl.maximum(y, 0.0)

    # softmax in fp32 (matches PyTorch bf16 softmax: fp32 accumulate, bf16 output)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    r = (e / s).to(tl.bfloat16)

    tl.store(out_ptr + row * D + offs, r, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(rows,)](
            x, self.b1, self.rms2_w, out, d, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
