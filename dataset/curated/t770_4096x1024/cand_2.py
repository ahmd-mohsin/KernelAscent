import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 770
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_bias_rms_ln_kernel(
    X_ptr, B0_ptr, W_ptr, G_ptr, B_ptr, Y_ptr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- load & bias add (bf16 add == fp32 add rounded once to bf16) ----
    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xb = (x + b0).to(tl.bfloat16)          # matches: x = x + self.b0 (bf16)
    xf = xb.to(tl.float32)                 # matches: _xf = x.float()

    # ---- RMS norm (fp32) ----
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    rinv = 1.0 / tl.sqrt(ms + 1e-6)
    xr16 = (xf * rinv).to(tl.bfloat16)     # matches: .to(x.dtype)

    # bf16 * bf16 multiply == fp32 multiply rounded once to bf16
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    xw16 = (xr16.to(tl.float32) * w).to(tl.bfloat16)  # matches: * self.rms1_w
    h = xw16.to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, eps=1e-5) ----
    mean = tl.sum(tl.where(mask, h, 0.0), axis=0) / D
    diff = tl.where(mask, h - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    tl.store(Y_ptr + row * D + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x + self.b0
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        rows = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_bias_rms_ln_kernel[(rows,)](
            xc, self.b0, self.rms1_w, self.ln2_g, self.ln2_b, y,
            D=d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
