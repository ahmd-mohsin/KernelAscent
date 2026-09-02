import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 583
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _fused_gelu_ln_relu_rms_kernel(
    X, G, B, W, Out,
    N,
    eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf), computed in fp32 like PyTorch, then rounded to fp16
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # LayerNorm (stats in fp32)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    d = tl.where(mask, g - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    gamma = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * gamma + beta
    y = y.to(tl.float16)

    # ReLU (on fp16 values)
    zero16 = tl.zeros_like(y)
    y = tl.maximum(y, zero16)

    # RMSNorm: stats in fp32, normalize, cast to fp16, then fp16 multiply by weight
    yf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + eps_rms)
    z = (yf * r).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    out = z * w

    tl.store(Out + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_relu_rms_kernel[(Mrows,)](
            h, self.ln2_g, self.ln2_b, self.rms4_w, out,
            N,
            1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
