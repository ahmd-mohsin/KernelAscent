import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 821
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_ln_softmax_ln(
    X, Y, G1, B1, G3, B3,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    x = tl.load(X + row * stride_x + offs).to(tl.float32)

    # LayerNorm 1
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + offs).to(tl.float32)
    b1 = tl.load(B1 + offs).to(tl.float32)
    y = d * rstd * g1 + b1
    # match bf16 rounding between ops
    y = y.to(tl.bfloat16).to(tl.float32)

    # Softmax
    mx = tl.max(y, axis=0)
    e = tl.exp(y - mx)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 3
    mean2 = tl.sum(p, axis=0) / N
    d2 = p - mean2
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + offs).to(tl.float32)
    b3 = tl.load(B3 + offs).to(tl.float32)
    out = d2 * rstd2 * g3 + b3

    tl.store(Y + row * stride_y + offs, out.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        _fused_ln_softmax_ln[(rows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            h.stride(0), out.stride(0),
            N=N, EPS=1e-5, BLOCK=512,
            num_warps=4,
        )
        return out
