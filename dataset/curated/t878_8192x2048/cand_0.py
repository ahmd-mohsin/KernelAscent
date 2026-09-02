import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 878
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, Y, G, B,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))

    # Softmax (fp32 accumulation, like PyTorch)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom

    # cast to bf16 (softmax output dtype) then back to fp32 for layernorm math
    s = s.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 math, eps=1e-5, biased variance)
    mean = tl.sum(tl.where(mask, s, 0.0), axis=0) / N
    diff = tl.where(mask, s - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b

    # cast to bf16 (layernorm output dtype), back to fp32 for gelu
    y = y.to(tl.bfloat16).to(tl.float32)

    # GELU (exact, erf)
    y = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))

    # cast to bf16 (gelu output), then scale in fp32 opmath, cast back
    y = y.to(tl.bfloat16).to(tl.float32)
    y = y * 1.1802

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x, y, self.ln2_g, self.ln2_b,
            x.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
