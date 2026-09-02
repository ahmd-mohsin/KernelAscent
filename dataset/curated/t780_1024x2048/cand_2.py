import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 780
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, G, B, W, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf), computed in fp32 then rounded to fp16 like PyTorch
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # scale, rounded to fp16
    g = (g * 1.0366).to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 accumulation, biased variance)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / D
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    ln = diff * inv_std * gamma + beta
    ln = ln.to(tl.float16).to(tl.float32)

    # scale, rounded to fp16
    ln = (ln * 1.4943).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32, cast to fp16, then multiply by weight (fp16-equivalent)
    ms = tl.sum(tl.where(mask, ln * ln, 0.0), axis=0) / D
    r = ln * tl.math.rsqrt(ms + 1e-6)
    r16 = r.to(tl.float16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (r16 * w).to(tl.float16)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float16 and x.shape[-1] == 2048:
            orig_shape = x.shape
            x2 = x.contiguous().view(-1, 2048)
            M_ = x2.shape[0]
            y = torch.empty_like(x2)
            BLOCK = 2048
            _fused_kernel[(M_,)](
                x2, self.ln2_g, self.ln2_b, self.rms4_w, y,
                D=2048, BLOCK=BLOCK,
                num_warps=8,
            )
            return y.view(orig_shape)

        # fallback (reference path)
        x = F.gelu(x)
        x = x * 1.0366
        x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
        x = x * 1.4943
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
        return x
