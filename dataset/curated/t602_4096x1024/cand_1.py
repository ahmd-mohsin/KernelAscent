import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 602
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_rms_gelu_kernel(
    X, W, B, Y,
    stride_xm, stride_ym,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)

    # RMS norm (fp32)
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = x * inv
    # round to bf16 (matches .to(x.dtype))
    xn = xn.to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    y = (xn * w).to(tl.bfloat16).to(tl.float32)   # bf16 mul rounds
    y = (y + b).to(tl.bfloat16).to(tl.float32)    # bf16 add rounds

    # exact GELU (erf), computed in fp32 then rounded to bf16
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    g = (g * 1.3699).to(tl.bfloat16).to(tl.float32)
    g = (g * 1.2531).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + offs, g, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_rms_gelu_kernel[(m,)](
            x, self.rms0_w, self.b1, y,
            x.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
