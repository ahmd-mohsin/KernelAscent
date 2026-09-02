import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 615
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _fused_gelu_ln_rms_kernel(
    X, OUT, G1, B1, W2,
    N, stride_x, stride_o,
    EPS_LN: tl.constexpr, EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then rounded to fp16 (matches PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    h = g.to(tl.float16).to(tl.float32)
    h = tl.where(mask, h, 0.0)

    # LayerNorm (fp32 accumulation, biased variance)
    mean = tl.sum(h, axis=0) / N
    diff = tl.where(mask, h - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + EPS_LN)

    w1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (h - mean) * rstd * w1 + b1
    y = y.to(tl.float16).to(tl.float32)  # round like layer_norm output in fp16
    y = tl.where(mask, y, 0.0)

    # RMSNorm in fp32, round to fp16, then scale by weight
    ms = tl.sum(y * y, axis=0) / N
    rrms = tl.math.rsqrt(ms + EPS_RMS)
    z = (y * rrms).to(tl.float16).to(tl.float32)

    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (z * w2).to(tl.float16)

    tl.store(OUT + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_rms_kernel[(rows,)](
            x2, out, self.ln1_g, self.ln1_b, self.rms2_w,
            N, x2.stride(0), out.stride(0),
            EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
