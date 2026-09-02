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
    X, Y,
    G1, B1, G2, B2, G3, B3,
    N: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    ptr = X + row * N + offs
    x = tl.load(ptr).to(tl.float32)

    # LayerNorm 1
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(G1 + offs).to(tl.float32)
    b = tl.load(B1 + offs).to(tl.float32)
    x = xc * rstd * w + b
    x = x.to(tl.bfloat16).to(tl.float32)  # match intermediate bf16 rounding

    # LayerNorm 2
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(G2 + offs).to(tl.float32)
    b = tl.load(B2 + offs).to(tl.float32)
    x = xc * rstd * w + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 3
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(G3 + offs).to(tl.float32)
    b = tl.load(B3 + offs).to(tl.float32)
    x = xc * rstd * w + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulation, bf16 output)
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * N + offs, y.to(tl.bfloat16))


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
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        _ln3_softmax_kernel[(rows,)](
            h, out,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.ln3_g, self.ln3_b,
            N=N, eps=1e-5,
            num_warps=4,
        )
        return out
