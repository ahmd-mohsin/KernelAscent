import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 907
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, w_ptr, out_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # relu
    xf = tl.maximum(xf, 0.0)

    # x * 1.0822 (bf16 rounding, matching PyTorch opmath float -> bf16)
    xf = (xf * 1.0822).to(tl.bfloat16).to(tl.float32)
    # x * 1.1211
    xf = (xf * 1.1211).to(tl.bfloat16).to(tl.float32)

    # RMS norm in float32
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(out_ptr + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        w = self.rms3_w
        if w.device != x.device:
            w = w.to(x.device)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](x2, w, out, d, BLOCK=BLOCK, num_warps=4)
        return out.view(orig_shape)
