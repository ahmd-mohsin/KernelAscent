import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 916
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _fused_ln_relu_rms_gelu(
    X, G, B, W,          # ptrs: input (M,N) fp16, ln gamma, ln beta, rms weight
    Y,                   # output (M,N) fp16
    N, stride,           # row length, row stride
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, biased variance)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    h = xc * rstd * g + b
    # cast to fp16 as PyTorch layer_norm outputs half, then relu
    h = h.to(tl.float16)
    h = tl.maximum(h, 0.0)

    # RMSNorm computed in fp32 on the half values
    hf = h.to(tl.float32)
    ms = tl.sum(tl.where(mask, hf * hf, 0.0), axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + RMS_EPS)

    t = (hf * rrms).to(tl.float16)  # matches .to(x.dtype)
    w = tl.load(W + cols, mask=mask, other=0.0)
    t = (t.to(tl.float32) * w.to(tl.float32)).to(tl.float16)  # half mul semantics

    # exact GELU (erf-based) in fp32
    tf = t.to(tl.float32)
    out = tf * 0.5 * (1.0 + tl.math.erf(tf * 0.7071067811865476))

    tl.store(Y + row * stride + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_relu_rms_gelu[(Mrows,)](
            x, self.ln1_g, self.ln1_b, self.rms3_w, y,
            N, x.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
