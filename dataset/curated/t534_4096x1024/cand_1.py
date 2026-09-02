import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 534
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_bias_softmax_scale_ln(
    X, B, G, Beta, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b16 = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in fp16 rounding semantics (fp32 add then round == fp16 add)
    z16 = (x16.to(tl.float32) + b16.to(tl.float32)).to(tl.float16)
    z = z16.to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax with float accumulation)
    z = tl.where(mask, z, float('-inf'))
    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    sm16 = sm.to(tl.float16)  # softmax output rounded to fp16

    # scale: half tensor * scalar -> computed in fp32 (opmath), rounded to fp16
    y16 = (sm16.to(tl.float32) * SCALE).to(tl.float16)
    y = y16.to(tl.float32)
    y = tl.where(mask, y, 0.0)

    # layer norm in fp32
    n = N
    mean = tl.sum(y, axis=0) / n
    diff = tl.where(mask, y - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(Beta + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y - mean) * rstd * g + beta

    tl.store(Out + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_softmax_scale_ln[(m,)](
            h, self.b1, self.ln4_g, self.ln4_b, out,
            h.stride(0), out.stride(0),
            N=n, EPS=1e-5, SCALE=1.0871,
            BLOCK=BLOCK, num_warps=8,
        )
        return out
