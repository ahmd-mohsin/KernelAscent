import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 379
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G, B, B3, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    # round to bf16 (matches PyTorch elementwise op output rounding)
    x = x.to(tl.bfloat16).to(tl.float32)

    # exact gelu: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    # round to bf16 before layernorm (matches PyTorch intermediate)
    x = x.to(tl.bfloat16).to(tl.float32)

    # layer norm (fp32 accumulation, biased variance)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    # layernorm output rounded to bf16
    y = y.to(tl.bfloat16).to(tl.float32)

    # add bias in fp32 opmath, round to bf16
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b3

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x + self.b3

        x = x.contiguous()
        rows = x.numel() // x.shape[-1]
        N = x.shape[-1]
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x, self.ln2_g, self.ln2_b, self.b3, y,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return y
