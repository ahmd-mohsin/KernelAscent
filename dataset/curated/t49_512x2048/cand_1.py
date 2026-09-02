import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 49
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(X, B, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0 (fp16 rounding as in reference)
    a = (x + b).to(tl.float16).to(tl.float32)

    # RMSNorm in float32
    ms = tl.sum(a * a, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    n = (a * r).to(tl.float16).to(tl.float32)

    # * rms1_w (fp16 round)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (n * w).to(tl.float16).to(tl.float32)

    # * 1.2274 (fp16 round)
    z = (y * 1.2274).to(tl.float16).to(tl.float32)

    # exact GELU (erf) computed in float32, output fp16
    g = z * 0.5 * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Y + row * D + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        _fused_kernel[(m,)](
            x, self.b0, self.rms1_w, out,
            D=d, BLOCK=triton.next_power_of_2(d),
            num_warps=8,
        )
        return out
