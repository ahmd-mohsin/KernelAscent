import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 537
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _fused_ln_ln_bias_gelu_relu(
    X, G1, B1, G2, B2, B3, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (compute in fp32, output rounded to bf16 like PyTorch)
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1
    y1 = y1.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(y1, axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = d2 * rstd2 * g2 + b2
    y2 = y2.to(tl.bfloat16).to(tl.float32)

    # + b3 (bf16 add rounding)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (y2.to(tl.bfloat16) + b3.to(tl.bfloat16)).to(tl.float32)

    # exact GELU (erf), rounded to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ReLU
    out = tl.maximum(g, 0.0)

    tl.store(Out + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = torch.matmul(x, self.W0)
        if not y.is_cuda:
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            y = y + self.b3
            return torch.relu(F.gelu(y))

        y = y.contiguous()
        Mrows, N = y.shape[0], y.shape[-1]
        y2d = y.view(-1, N)
        out = torch.empty_like(y2d)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_ln_bias_gelu_relu[(y2d.shape[0],)](
            y2d, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.b3, out,
            y2d.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view_as(y)
