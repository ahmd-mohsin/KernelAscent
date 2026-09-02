import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 593
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b2_ptr, w_ptr, out_ptr, N, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    offs = row * D + cols

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu 1 (compute in fp32, round to fp16 like PyTorch half kernels)
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # gelu 2
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # add bias (fp32 compute, round to fp16)
    xf = x.to(tl.float32) + b2.to(tl.float32)
    x = xf.to(tl.float16)

    # RMSNorm (fp32)
    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.float16)

    # multiply by weight (fp32 compute, round to fp16), then relu
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    y = tl.maximum(y, tl.zeros_like(y))

    tl.store(out_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2d = x.view(-1, Dcols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(n_rows,)](
            x2d, self.b2, self.rms3_w, out,
            n_rows, Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view_as(x)
