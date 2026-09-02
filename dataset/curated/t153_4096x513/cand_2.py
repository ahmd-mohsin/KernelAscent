import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 153
M, D, DT = 4096, 513, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, W, G, B, OUT,
    n_cols,
    stride_x, stride_o,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (float32 math, cast to bf16, multiply by bf16 weight) ----
    ms = tl.sum(x * x, axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    xn_bf = (x * r).to(tl.bfloat16)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.bfloat16)
    y_bf = xn_bf * w  # bf16 multiply, matching eager

    # ---- Softmax (accumulate in float32, output rounded to bf16) ----
    yf = tl.where(mask, y_bf.to(tl.float32), float('-inf'))
    mx = tl.max(yf, axis=0)
    e = tl.exp(yf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p_bf = (e / s).to(tl.bfloat16)

    # ---- LayerNorm (float32 math on bf16 input) ----
    pf = p_bf.to(tl.float32)
    pf = tl.where(mask, pf, 0.0)
    mean = tl.sum(pf, axis=0) / n_cols
    diff = tl.where(mask, pf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    ln_bf = (diff * inv * g + b).to(tl.bfloat16)

    # ---- Scale (opmath float32, output bf16) ----
    out = (ln_bf.to(tl.float32) * SCALE).to(tl.bfloat16)
    tl.store(OUT + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x * 1.3856

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.contiguous().view(-1, n_cols)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_kernel[(n_rows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, out,
            n_cols,
            x2.stride(0), out.stride(0),
            EPS_RMS=1e-6,
            EPS_LN=1e-5,
            SCALE=1.3856,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
