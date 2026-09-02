import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 170
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_bias_softmax_gelu_ln(
    X, B, G, Bt, Out,
    stride_xm, stride_om,
    N, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # bias add (fp16 rounding like reference)
    t = (x + b).to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, cast to fp16 like reference output)
    t_masked = tl.where(mask, t, float('-inf'))
    mx = tl.max(t_masked, axis=0)
    e = tl.exp(t_masked - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # exact GELU (erf-based), rounded to fp16 like reference
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = (sm * 0.5 * (1.0 + tl.math.erf(sm * INV_SQRT2))).to(tl.float16).to(tl.float32)
    g = tl.where(mask, g, 0.0)

    # layernorm in fp32
    mean = tl.sum(g, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Bt + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * gamma + beta

    tl.store(Out + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM with tensor cores
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_bias_softmax_gelu_ln[(Mrows,)](
            h, self.b1, self.ln4_g, self.ln4_b, out,
            h.stride(0), out.stride(0),
            N, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
