import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 797
M, D, DT = 8192, 4097, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, Y,
    G1, B1, W2, G3, B3, W4,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    # ReLU
    x = tl.maximum(x, 0.0)

    # ---- LayerNorm 1 (fp32 compute, bf16 output) ----
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    inv1 = tl.math.rsqrt(var1 + 1e-5)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (d1 * inv1 * g1 + b1).to(tl.bfloat16)

    # ---- RMSNorm 2 ----
    yf = y.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    z = (yf * tl.math.rsqrt(ms2 + 1e-6)).to(tl.bfloat16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z.to(tl.float32) * w2).to(tl.bfloat16)

    # ---- LayerNorm 3 ----
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    mean3 = tl.sum(zf, axis=0) / N
    d3 = tl.where(mask, zf - mean3, 0.0)
    var3 = tl.sum(d3 * d3, axis=0) / N
    inv3 = tl.math.rsqrt(var3 + 1e-5)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    u = (d3 * inv3 * g3 + b3).to(tl.bfloat16)

    # ---- RMSNorm 4 ----
    uf = u.to(tl.float32)
    ms4 = tl.sum(tl.where(mask, uf * uf, 0.0), axis=0) / N
    v = (uf * tl.math.rsqrt(ms4 + 1e-6)).to(tl.bfloat16)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (v.to(tl.float32) * w4).to(tl.bfloat16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.reshape(-1, N)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_norm_kernel[(rows,)](
            x2, out,
            self.ln1_g, self.ln1_b, self.rms2_w,
            self.ln3_g, self.ln3_b, self.rms4_w,
            N, x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=32,
        )
        return out.reshape(orig_shape)
