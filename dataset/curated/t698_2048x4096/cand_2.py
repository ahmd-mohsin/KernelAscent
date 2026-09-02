import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 698
M, D, DT = 2048, 4096, torch.bfloat16

_INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_gelu_rms_gelu(X, W, Y, D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # gelu #1 (exact erf gelu, fp32 math, cast back to bf16 like PyTorch)
    g1 = 0.5 * xf * (1.0 + tl.math.erf(xf * _INV_SQRT2))
    g1_bf = g1.to(tl.bfloat16)

    # RMS norm in fp32
    gf = g1_bf.to(tl.float32)
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + EPS)
    normed = (gf * inv).to(tl.bfloat16)

    # scale by weight (bf16 mul with fp32 opmath -> bf16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    scaled = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # gelu #2
    sf = scaled.to(tl.float32)
    g2 = 0.5 * sf * (1.0 + tl.math.erf(sf * _INV_SQRT2))
    y = g2.to(tl.bfloat16)

    tl.store(Y + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.gelu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        y = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        _fused_gelu_rms_gelu[(m,)](
            xc, self.rms1_w, y,
            D=d, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
