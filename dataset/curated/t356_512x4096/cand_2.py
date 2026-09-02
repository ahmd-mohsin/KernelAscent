import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 356
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _gelu_f32(x):
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))


@triton.jit
def _round_bf16(x):
    return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _fused_kernel(
    X, W, B, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu 1 (rounded to bf16 like the eager op boundary)
    x = _round_bf16(_gelu_f32(x))

    # layer norm (fp32 accumulation)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * w + b
    y = _round_bf16(y)

    # gelu 2, 3, 4
    y = _round_bf16(_gelu_f32(y))
    y = _round_bf16(_gelu_f32(y))
    y = _gelu_f32(y)

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        M_, N = x.shape[0], x.shape[-1]
        x2d = x.view(-1, N)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2d, self.ln1_g, self.ln1_b, y,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view_as(x)
