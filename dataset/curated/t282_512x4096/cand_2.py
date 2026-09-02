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
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 math, round to bf16 like eager between ops)
    m = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - m, 0.0)
    v = tl.sum(xc * xc, axis=0) / N
    x = xc * tl.rsqrt(v + 1e-5) * g1 + b1
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    m = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - m, 0.0)
    v = tl.sum(xc * xc, axis=0) / N
    x = xc * tl.rsqrt(v + 1e-5) * g2 + b2
    x = x.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 3
    m = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - m, 0.0)
    v = tl.sum(xc * xc, axis=0) / N
    x = xc * tl.rsqrt(v + 1e-5) * g3 + b3
    x = x.to(tl.bfloat16).to(tl.float32)

    # Softmax
    x = tl.where(mask, x, float('-inf'))
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


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
        h = torch.matmul(x, self.W0)  # (M, 512) bf16 via tensor cores
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        _ln3_softmax_kernel[(rows,)](
            h, out,
            self.ln1_g, self.ln1_b,
            self.ln2_g, self.ln2_b,
            self.ln3_g, self.ln3_b,
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=4,
        )
        return out
