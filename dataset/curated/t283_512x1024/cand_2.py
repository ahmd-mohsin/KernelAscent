import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 283
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, B, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf) computed in fp32, cast back to bf16 (matches PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # ReLU in bf16
    g = tl.maximum(g, tl.zeros_like(g))

    # Softmax in fp32 (matches PyTorch bf16 softmax with float accumulation)
    xf = g.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # Add bias (fp32 opmath, cast back to bf16)
    b = tl.load(B + offs, mask=mask, other=0.0)
    out = (sm.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # Scale (fp32 opmath, cast back to bf16)
    out = (out.to(tl.float32) * 1.2457).to(tl.bfloat16)

    tl.store(Y + row * D_ + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](x, self.b3, y, d, BLOCK=BLOCK, num_warps=8)
        return y
