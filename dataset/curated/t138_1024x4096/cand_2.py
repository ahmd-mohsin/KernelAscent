import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 138
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_scale_relu_rms_kernel(
    X, W, Y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0)
    # multiply in fp32, round back to bf16 (matches PyTorch bf16 elementwise op)
    xf = x.to(tl.float32) * 1.0731
    xb = xf.to(tl.bfloat16)
    # relu in bf16 (exact, no rounding needed)
    xb = tl.maximum(xb, 0.0)

    # RMS norm in fp32
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = tl.math.rsqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    # bf16 * bf16 -> compute in fp32, round to bf16 (matches PyTorch)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    tl.store(Y + row * D_ + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows = x.numel() // x.shape[-1]
        d = x.shape[-1]
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_scale_relu_rms_kernel[(rows,)](
            x, self.rms2_w, y,
            d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
