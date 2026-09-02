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
    X, G1, B1, G2, B2, B3, Y,
    N: tl.constexpr, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 math, matching PyTorch's bf16 layer_norm behavior)
    mu = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mu, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)
    y = d * inv
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = y * g1 + b1
    # round to bf16 like the reference (output of first layer_norm is bf16)
    y = y.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mu2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d2 = tl.where(mask, y - mu2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(var2 + eps)
    y2 = d2 * inv2
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y2 = y2 * g2 + b2
    y2 = y2.to(tl.bfloat16).to(tl.float32)

    # bias add (bf16 rounding to match x + b3 in bf16)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    z = y2 + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf) then ReLU
    g = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
    g = tl.maximum(g, 0.0)

    tl.store(Y + row * N + offs, g.to(tl.bfloat16), mask=mask)


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
        # matmul via cuBLAS tensor cores (bf16)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_ln_bias_gelu_relu[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.b3, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
