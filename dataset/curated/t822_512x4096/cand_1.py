import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 822
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_gelu_ln_relu_ln(X, Y, G2, B2, G4, B4, N, stride_x, stride_y,
                           EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32, rounded to bf16 like PyTorch
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    d = tl.where(mask, g - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    h = d * rstd * g2 + b2
    h = h.to(tl.bfloat16).to(tl.float32)

    # ReLU
    h = tl.maximum(h, 0.0)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, h, 0.0), axis=0) / N
    d2 = tl.where(mask, h - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g4 + b4

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_ln_relu_ln[(m,)](
            h, y, self.ln2_g, self.ln2_b, self.ln4_g, self.ln4_b,
            n, h.stride(0), y.stride(0),
            EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
