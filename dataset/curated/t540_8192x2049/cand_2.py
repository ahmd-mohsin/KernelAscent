import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 540
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _fused_kernel(X, W, B1, B2, Y, D,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- RMSNorm (fp32 math, matches reference) ----
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16)  # cast to bf16 like reference

    # ---- scale + biases in bf16 (matches reference rounding) ----
    w = tl.load(W + offs, mask=mask, other=0.0)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)
    v = ((xn * w) + b1) + b2  # bf16 arithmetic

    # ---- exact (erf) GELU in fp32, cast back to bf16 ----
    vf = v.to(tl.float32)
    g = vf * 0.5 * (1.0 + tl.math.erf(vf * 0.7071067811865476))
    gb = g.to(tl.bfloat16).to(tl.float32)

    # ---- softmax in fp32, output bf16 ----
    gb = tl.where(mask, gb, float('-inf'))
    mx = tl.max(gb, axis=0)
    e = tl.exp(gb - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_kernel[(Mrows,)](
            x, self.rms0_w, self.b1, self.b2, y, Dcols,
            BLOCK=BLOCK, num_warps=8,
        )
        return y
