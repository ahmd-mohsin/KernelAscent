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

    # ---- LayerNorm 0 (fp32 accumulation, output rounded to bf16) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + EPS_LN)
    g0 = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g0 + b0
    y = y.to(tl.bfloat16)

    # ---- scalar multiply (opmath fp32, round to bf16) ----
    y = (y.to(tl.float32) * SCALE).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = tl.math.rsqrt(var2 + EPS_LN)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    z = (d2 * rstd2 * g2 + b2).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm ----
    ms = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + EPS_RMS)
    w = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = ((z * r).to(tl.bfloat16).to(tl.float32) * w).to(tl.bfloat16)

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
            return self._forward_ref(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_ln_ln_rms_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, self.ln2_g, self.ln2_b, self.rms3_w, y,
            N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            SCALE=1.3978,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
        x = x * 1.3978
        x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        return x
