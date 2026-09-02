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
    X, OUT,
    G0, B0, G1, B1, W3,
    N, stride_x, stride_o,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, bf16 round-trip like PyTorch) ----
    mean0 = tl.sum(x, axis=0) / N
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + EPS_LN)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y0 = d0 * rstd0 * g0 + b0
    y0 = y0.to(tl.bfloat16).to(tl.float32)  # round to bf16 as PyTorch does between ops

    # ---- LayerNorm 1 ----
    y0 = tl.where(mask, y0, 0.0)
    mean1 = tl.sum(y0, axis=0) / N
    d1 = tl.where(mask, y0 - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS_LN)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1
    y1 = y1.to(tl.bfloat16).to(tl.float32)

    # ---- ReLU (bf16 exact, no rounding change) ----
    y1 = tl.maximum(y1, 0.0)
    y1 = tl.where(mask, y1, 0.0)

    # ---- RMSNorm ----
    ms = tl.sum(y1 * y1, axis=0) / N
    scale = 1.0 / tl.sqrt(ms + EPS_RMS)
    z = (y1 * scale).to(tl.bfloat16).to(tl.float32)  # (_xf*rsqrt).to(bf16)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * w3).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


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
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.relu(y)
            _xf = y.float()
            return (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_ln_relu_rms_kernel[(rows,)](
            x2, out,
            self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b, self.rms3_w,
            N, x2.stride(0), out.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
