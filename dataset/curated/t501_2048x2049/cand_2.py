import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 501
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _fused_ln_ln_relu_rms_kernel(
    x_ptr, out_ptr,
    g0_ptr, b0_ptr, g1_ptr, b1_ptr, w3_ptr,
    N, x_stride, o_stride,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 accumulate, bf16 round like PyTorch) ----
    mean0 = tl.sum(x, axis=0) / N
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / N
    rstd0 = tl.math.rsqrt(var0 + EPS_LN)
    g0 = tl.load(g0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d0 * rstd0 * g0 + b0
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 1 ----
    y = tl.where(mask, y, 0.0)
    mean1 = tl.sum(y, axis=0) / N
    d1 = tl.where(mask, y - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = tl.math.rsqrt(var1 + EPS_LN)
    g1 = tl.load(g1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = d1 * rstd1 * g1 + b1
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- ReLU (exact on bf16 values) ----
    z = tl.maximum(z, 0.0)

    # ---- RMSNorm: fp32 math, cast to bf16, then bf16 multiply by weight ----
    z = tl.where(mask, z, 0.0)
    ms = tl.sum(z * z, axis=0) / N
    rrms = tl.math.rsqrt(ms + EPS_RMS)
    zn = (z * rrms).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(w3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (zn * w3).to(tl.bfloat16)

    tl.store(out_ptr + row * o_stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.relu(y)
            _yf = y.float()
            y = (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_ln_relu_rms_kernel[(Mrows,)](
            x2, out,
            self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b, self.rms3_w,
            N, x2.stride(0), out.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
