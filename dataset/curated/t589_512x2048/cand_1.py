import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 589
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_ln_bias_relu_gelu2(
    X, G, B, B2, Out,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm stats in fp32 (matches PyTorch bf16 layer_norm behavior)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    # round to bf16 (layer_norm output dtype)
    y = y.to(tl.bfloat16).to(tl.float32)

    # + b2, round to bf16
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b2).to(tl.bfloat16).to(tl.float32)

    # relu (exact on bf16 values)
    y = tl.maximum(y, 0.0)

    # gelu (exact erf), computed in fp32, rounded to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = (0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    # second gelu
    y = (0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))).to(tl.bfloat16)

    tl.store(Out + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # bf16 matmul (tensor cores), identical to reference
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_ln_bias_relu_gelu2[(m,)](
            y, self.ln1_g, self.ln1_b, self.b2, out,
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
