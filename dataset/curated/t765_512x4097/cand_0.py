import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 765
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X, B0, G, B1, W, Y,
    D: tl.constexpr,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0  (bf16 elementwise add semantics: fp32 add, round to bf16)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 internal math, like PyTorch for bf16 inputs)
    mean = tl.sum(x, axis=0) / D
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D
    inv_std = tl.math.rsqrt(var + LN_EPS)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * inv_std * g + b1
    # cast back to bf16 (layer_norm output dtype), then re-promote for RMS
    y = y.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / D
    rrms = tl.math.rsqrt(ms + RMS_EPS)
    y = (y * rrms).to(tl.bfloat16).to(tl.float32)

    # multiply by rms2_w in bf16 semantics (fp32 mul, round to bf16)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_ln_rms_kernel[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.rms2_w, out,
            D=d,
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out.view(orig_shape)
