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
    X, B0, G1, B1, W2, OUT,
    N, stride_x, stride_o,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # x = x + b0 in bf16 (round to bf16 like the eager op)
    a_bf = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.bfloat16)
    a = a_bf.to(tl.float32)
    a = tl.where(mask, a, 0.0)

    # LayerNorm in fp32 (as PyTorch does internally for bf16 inputs)
    n_f = N.to(tl.float32)
    mean = tl.sum(a, axis=0) / n_f
    diff = tl.where(mask, a - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_f
    inv_std = 1.0 / tl.sqrt(var + EPS_LN)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv_std * g1 + b1
    # cast to bf16 (layer_norm output dtype), then back to fp32 for RMS step
    y_bf = y.to(tl.bfloat16)
    yf = y_bf.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)

    # RMSNorm in fp32
    ms = tl.sum(yf * yf, axis=0) / n_f
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)
    z_bf = (yf * rrms).to(tl.bfloat16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    out = (z_bf.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + row * stride_o + cols, out, mask=mask)


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
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_ln_rms_kernel[(Mrows,)](
            x2d, self.b0, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N, x2d.stride(0), out.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=16,
        )
        return out.view(orig_shape)
