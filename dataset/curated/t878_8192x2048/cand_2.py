import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 878
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_relu_softmax_ln_gelu_kernel(
    X_ptr, Y_ptr, G_ptr, B_ptr,
    n_cols, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X_ptr + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # Softmax (fp32 accumulation, matching PyTorch's AccumulateType)
    x_for_max = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x_for_max, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom
    # round through bf16 as PyTorch does between ops
    p = p.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation from bf16 input)
    n = n_cols.to(tl.float32)
    mean = tl.sum(p, axis=0) / n
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (p - mean) * rstd * g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # GELU (exact, erf-based) in fp32 opmath, then scale
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y * scale

    tl.store(Y_ptr + row * n_cols + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_relu_softmax_ln_gelu_kernel[(n_rows,)](
            x, y, self.ln2_g, self.ln2_b,
            n_cols, 1e-5, 1.1802,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
