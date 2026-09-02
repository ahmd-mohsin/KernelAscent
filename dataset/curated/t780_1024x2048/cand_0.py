import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 780
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_gelu_ln_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    n_cols,
    stride_x, stride_y,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU (exact, erf) computed in fp32, rounded to fp16 like PyTorch ----
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # ---- scalar mul (opmath fp32, round to fp16) ----
    g = (g16.to(tl.float32) * 1.0366)
    g = g.to(tl.float16).to(tl.float32)

    # ---- LayerNorm (fp32 stats, fp16 output) ----
    gm = tl.where(mask, g, 0.0)
    mean = tl.sum(gm, axis=0) / n_cols
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    ln_g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    ln_b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * ln_g + ln_b
    y16 = y.to(tl.float16)

    # ---- scalar mul (opmath fp32, round to fp16) ----
    y = (y16.to(tl.float32) * 1.4943)
    y = y.to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32), cast to fp16, multiply by weight ----
    ym = tl.where(mask, y, 0.0)
    ms = tl.sum(ym * ym, axis=0) / n_cols
    rrms = 1.0 / tl.sqrt(ms + RMS_EPS)
    out16 = (y * rrms).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = (out16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            y = F.gelu(x) * 1.0366
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b) * 1.4943
            _yf = y.float()
            return (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms4_w

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]

        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_ln_rms_kernel[(n_rows,)](
            x2d, self.ln2_g, self.ln2_b, self.rms4_w, out,
            n_cols,
            x2d.stride(0), out.stride(0),
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
