import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 262
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_ln_ln_rms_kernel(
    X, G0, B0, G2, B2, W3, Y,
    N,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, bf16 output like PyTorch) ----
    mean0 = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(xc * xc, axis=0) / N
    inv0 = tl.math.rsqrt(var0 + EPS_LN)
    g0 = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xc * inv0 * g0 + b0).to(tl.bfloat16)

    # ---- x * 1.3978 (opmath fp32, result bf16) ----
    y = (y.to(tl.float32) * SCALE).to(tl.bfloat16)

    # ---- LayerNorm 2 ----
    x2 = y.to(tl.float32)
    mean2 = tl.sum(tl.where(mask, x2, 0.0), axis=0) / N
    x2c = tl.where(mask, x2 - mean2, 0.0)
    var2 = tl.sum(x2c * x2c, axis=0) / N
    inv2 = tl.math.rsqrt(var2 + EPS_LN)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    z = (x2c * inv2 * g2 + b2).to(tl.bfloat16)

    # ---- RMSNorm 3 ----
    zf = z.to(tl.float32)
    ms = tl.sum(tl.where(mask, zf * zf, 0.0), axis=0) / N
    rinv = tl.math.rsqrt(ms + EPS_RMS)
    zn = (zf * rinv).to(tl.bfloat16)  # cast to dtype first, like reference
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (zn.to(tl.float32) * w3).to(tl.bfloat16)

    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = x * 1.3978
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_ln_rms_kernel[(rows,)](
            x2d, self.ln0_g, self.ln0_b, self.ln2_g, self.ln2_b, self.rms3_w, y,
            N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            SCALE=1.3978,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
