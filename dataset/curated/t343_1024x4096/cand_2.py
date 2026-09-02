import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 343
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0)
    # relu (applied twice == once), then float
    xf = tl.maximum(x, 0.0).to(tl.float32)

    # rms norm in fp32
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    # normalize, round to fp16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    # fp16*fp16 elementwise: PyTorch computes in fp32 (opmath), rounds to fp16
    xm = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # relu
    xr = tl.maximum(xm, 0.0)

    # scalar mul: computed in fp32, rounded to fp16
    out = (xr.to(tl.float32) * 1.1395).to(tl.float16)

    tl.store(Y + row * D_ + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, d = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.rms2_w, y, d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
