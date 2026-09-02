import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 399
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_bias_ln_rms_kernel(
    X, B, G, Beta, W, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B + cols, mask=mask, other=0.0)                    # fp16

    # bias add in fp16 (matches x + b1 in half precision)
    v16 = x + b
    v = v16.to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch half layer_norm internals)
    mean = tl.sum(tl.where(mask, v, 0.0), axis=0) / N
    diff = tl.where(mask, v - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta + cols, mask=mask, other=0.0).to(tl.float32)
    y32 = (v - mean) * rstd * g + beta
    y16 = y32.to(tl.float16)  # round to fp16 (layer_norm output dtype)

    # RMSNorm: compute in fp32 from the fp16 layernorm output
    yf = y16.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS_RMS)
    z16 = (yf * inv).to(tl.float16)  # .to(x.dtype)

    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    out = z16 * w  # fp16 multiply, matches reference

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_bias_ln_rms_kernel[(m,)](
            h, self.b1, self.ln2_g, self.ln2_b, self.rms3_w, out,
            h.stride(0), out.stride(0),
            N=n, EPS_LN=1e-5, EPS_RMS=1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
