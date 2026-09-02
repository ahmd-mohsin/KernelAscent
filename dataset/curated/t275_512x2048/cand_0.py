import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 275
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_rms_ln_gelu(
    X, W0, G, B, Y,
    D: tl.constexpr,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm (computed in fp32, rounded to bf16 like reference) ----
    ms = tl.sum(x * x, axis=0) / D
    inv_rms = tl.math.rsqrt(ms + EPS_RMS)
    xn = (x * inv_rms).to(tl.bfloat16).to(tl.float32)

    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    # bf16 multiply (single rounding of fp32 product == bf16*bf16 round-nearest)
    x1 = (xn * w0).to(tl.bfloat16).to(tl.float32)
    x1 = tl.where(mask, x1, 0.0)

    # ---- LayerNorm (fp32 accumulation, matches PyTorch acc type) ----
    mean = tl.sum(x1, axis=0) / D
    diff = tl.where(mask, x1 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = tl.math.rsqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv_std * g + b
    # round to bf16 (layer_norm output dtype) before gelu, like reference
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- GELU (exact, erf-based, fp32 opmath like PyTorch) ----
    out = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * D + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return F.gelu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_rms_ln_gelu[(n_rows,)](
            x2, self.rms0_w, self.ln1_g, self.ln1_b, y,
            D=d, EPS_RMS=1e-6, EPS_LN=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
