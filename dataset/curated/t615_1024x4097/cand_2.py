import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 615
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_gelu_ln_rms_kernel(
    X, G, B, W, Y,
    D, stride,
    eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU, computed in fp32 then rounded to fp16 (matches PyTorch half gelu)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, 0.0)

    # LayerNorm (stats in fp32, matches PyTorch layer_norm on fp16)
    Df = D.to(tl.float32)
    mean = tl.sum(gf, axis=0) / Df
    diff = tl.where(mask, gf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / Df
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    gamma = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    ln = diff * rstd * gamma + beta
    ln16 = ln.to(tl.float16)

    # RMSNorm computed explicitly in fp32 (matches reference)
    xf = ln16.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / Df
    rrms = 1.0 / tl.sqrt(ms + eps_rms)
    out16 = (xf * rrms).to(tl.float16)

    # final scale performed in fp16 (matches fp16 * fp16 elementwise mul)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = out16 * w
    tl.store(Y + row * stride + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 16 if BLOCK >= 8192 else 8

        _fused_gelu_ln_rms_kernel[(m,)](
            x2, self.ln1_g, self.ln1_b, self.rms2_w, y,
            d, x2.stride(0),
            1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
