import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 282
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _ln3_softmax_kernel(
    X, OUT,
    G1, B1, G2, B2, G3, B3,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)

    nf = N.to(tl.float32)

    # LayerNorm 1 (stats in fp32, output rounded to bf16 like PyTorch)
    mean = tl.sum(x, axis=0) / nf
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / nf
    rstd = 1.0 / tl.sqrt(var + EPS)
    y = d * rstd * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / nf
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / nf
    rstd = 1.0 / tl.sqrt(var + EPS)
    y = d * rstd * g2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 3
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / nf
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / nf
    rstd = 1.0 / tl.sqrt(var + EPS)
    y = d * rstd * g3 + b3
    y = y.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulation, bf16 output)
    y = tl.where(mask, y, float("-inf"))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln3_softmax_kernel[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.ln3_g, self.ln3_b,
            N, h.stride(0), out.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
