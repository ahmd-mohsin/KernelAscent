import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 698
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_gelu_rms_gelu(
    x_ptr, w_ptr, out_ptr,
    D: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based) computed in fp32 then cast back to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16)

    # RMSNorm in fp32
    xf = g.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = tl.math.rsqrt(ms + EPS)
    y = (xf * inv).to(tl.bfloat16)

    # scale by weight (bf16 * bf16)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.bfloat16)
    y = y * w

    # second GELU in fp32, cast back to bf16
    yf = y.to(tl.float32)
    g2 = yf * 0.5 * (1.0 + tl.math.erf(yf * INV_SQRT2))
    out = g2.to(tl.bfloat16)

    tl.store(out_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xg = F.gelu(x)
            _xf = xg.float()
            xg = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xg.dtype) * self.rms1_w
            return F.gelu(xg)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_gelu_rms_gelu[(m,)](
            x2, self.rms1_w, out,
            D=d, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
