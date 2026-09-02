import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 640
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_ln_gelu(
    X, W, B, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch CUDA softmax on bf16)
    row_max = tl.max(x, axis=0)
    ex = tl.exp(x - row_max)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    s = ex / denom
    # round to bf16 (softmax output dtype), then upcast for layernorm
    s = s.to(tl.bfloat16).to(tl.float32)
    s = tl.where(mask, s, 0.0)

    # layernorm in fp32
    n_f = N.to(tl.float32)
    mean = tl.sum(s, axis=0) / n_f
    d = tl.where(mask, s - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * w + b
    # round to bf16 (layernorm output dtype), then upcast for gelu
    y = y.to(tl.bfloat16).to(tl.float32)

    # gelu (erf-based, fp32 opmath)
    g = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 16 if BLOCK >= 8192 else 8
        _fused_softmax_ln_gelu[(Mrows,)](
            x, self.ln1_g, self.ln1_b, y,
            N, x.stride(0), y.stride(0),
            EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
