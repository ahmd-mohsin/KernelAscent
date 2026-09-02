import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 675
M, D, DT = 1024, 2048, torch.bfloat16

INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_kernel(
    x_ptr, w3_ptr, w4_ptr, out_ptr,
    D: tl.constexpr, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # gelu (exact, erf-based), computed in fp32, rounded to bf16
    g1 = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g1 = g1.to(tl.bfloat16).to(tl.float32)

    # scale by 1.113, rounded to bf16
    y = (g1 * scale).to(tl.bfloat16).to(tl.float32)

    # second gelu
    g2 = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # rmsnorm 1 (stats in fp32, normalized value cast to bf16, then * w3 in fp32)
    ms1 = tl.sum(tl.where(mask, g2 * g2, 0.0), axis=0) / D
    r1 = tl.math.rsqrt(ms1 + eps)
    n1 = (g2 * r1).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    h = (n1 * w3).to(tl.bfloat16).to(tl.float32)

    # rmsnorm 2
    ms2 = tl.sum(tl.where(mask, h * h, 0.0), axis=0) / D
    r2 = tl.math.rsqrt(ms2 + eps)
    n2 = (h * r2).to(tl.bfloat16).to(tl.float32)
    w4 = tl.load(w4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (n2 * w4).to(tl.bfloat16)

    tl.store(out_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x * 1.113
            x = F.gelu(x)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        out = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            xc, self.rms3_w, self.rms4_w, out,
            d, 1e-6, 1.113,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
