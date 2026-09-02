import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 641
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, rounded to bf16 like reference)
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16)

    # multiply by weight in bf16 precision
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # gelu (exact, erf) twice: compute in fp32, round to bf16 each time
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    yf = y.to(tl.float32)
    g1 = (0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))).to(tl.bfloat16)
    g1f = g1.to(tl.float32)
    g2 = (0.5 * g1f * (1.0 + tl.math.erf(g1f * INV_SQRT2))).to(tl.bfloat16)

    # softmax in fp32 accumulation
    z = g2.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = F.gelu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](x2, self.rms0_w, y, d, BLOCK=BLOCK,
                            num_warps=4 if BLOCK <= 1024 else 8)
        return y.view(orig_shape)
