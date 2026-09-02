import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 857
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_ln_softmax3_kernel(
    X, G, B, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, output rounded to fp16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # ---- Softmax 1 ----
    m1 = tl.max(tl.where(mask, y, float('-inf')), axis=0)
    e1 = tl.exp(y - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- Softmax 2 ----
    m2 = tl.max(tl.where(mask, y, float('-inf')), axis=0)
    e2 = tl.exp(y - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y = (e2 / s2).to(tl.float16).to(tl.float32)

    # ---- Scale (fp32 opmath, round to fp16 like PyTorch) ----
    y = (y * SCALE).to(tl.float16).to(tl.float32)

    # ---- Softmax 3 ----
    m3 = tl.max(tl.where(mask, y, float('-inf')), axis=0)
    e3 = tl.exp(y - m3)
    e3 = tl.where(mask, e3, 0.0)
    s3 = tl.sum(e3, axis=0)
    out = (e3 / s3).to(tl.float16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_softmax3_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, out,
            h.stride(0), out.stride(0),
            N=N, EPS=1e-5, SCALE=1.3209, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
