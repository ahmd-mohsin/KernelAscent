import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 439
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based)
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))
    # emulate bf16 rounding of gelu output (PyTorch computes gelu in bf16)
    g = g.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation)
    gv = tl.where(mask, g, 0.0)
    mean = tl.sum(gv, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    w = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g - mean) * rstd * w + b
    # layer_norm output is bf16
    y = y.to(tl.bfloat16).to(tl.float32)

    # ReLU
    y = tl.maximum(y, 0.0)
    # relu output in bf16 (exact anyway)

    # Softmax (fp32)
    y_masked = tl.where(mask, y, float('-inf'))
    m = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    # softmax output cast to bf16, then scale in bf16
    sm = sm.to(tl.bfloat16).to(tl.float32)
    out = sm * 1.4972

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(Mrows,)](
            x, self.ln1_g, self.ln1_b, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
