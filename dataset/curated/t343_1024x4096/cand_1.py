import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 343
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(x_ptr, w_ptr, out_ptr, D: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    # relu (applied twice == once)
    x = tl.maximum(x, 0.0)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + eps)

    # normalize in fp32, cast to fp16, then multiply by weight in fp16 (matches reference)
    y = (xf * inv).to(tl.float16)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = y * w
    # relu
    y = tl.maximum(y, tl.full((1,), 0.0, tl.float16))
    # scale in fp16 (torch: half tensor * python float -> half arithmetic)
    scale = tl.full((1,), 1.1395, tl.float16)
    y = y * scale

    tl.store(out_ptr + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.rms2_w, out, Dcols, 1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
