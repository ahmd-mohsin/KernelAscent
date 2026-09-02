import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 138
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(X, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0)  # bf16
    # x * 1.0731 computed in fp32, rounded back to bf16 (PyTorch semantics)
    xf = x.to(tl.float32) * 1.0731
    xb = xf.to(tl.bfloat16)
    # relu
    xb = tl.maximum(xb, 0.0)
    # rmsnorm in fp32
    f = xb.to(tl.float32)
    ms = tl.sum(f * f, axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    y = (f * inv).to(tl.bfloat16)
    # multiply by weight (fp32 compute, bf16 result)
    w = tl.load(W + offs, mask=mask, other=0.0)
    out = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.rms2_w, y, Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
