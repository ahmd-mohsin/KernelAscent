import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 610
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_ln_bias_gelu_relu_kernel(
    X, G, B, B2, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm statistics in fp32 (matches PyTorch bf16 layer_norm accumulation)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * g + b
    # layer_norm output is rounded to bf16 before subsequent ops
    y = y.to(tl.bfloat16).to(tl.float32)

    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU computed in fp32, as PyTorch does for bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gel = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    gel = gel.to(tl.bfloat16).to(tl.float32)

    out = tl.maximum(gel, 0.0)
    tl.store(Y + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core matmul
        h = x @ self.W0

        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_bias_gelu_relu_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, self.b2, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
