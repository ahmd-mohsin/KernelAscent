import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 336
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_gelu_bias_softmax(
    x_ptr, b1_ptr, b2_ptr, out_ptr,
    n_cols,
    x_stride, out_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32, rounded to bf16 like PyTorch
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16)

    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)

    # sequential bf16 adds to match reference rounding
    y = (g + b1).to(tl.bfloat16)
    y = (y + b2).to(tl.bfloat16)

    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float("-inf"))
    row_max = tl.max(yf, axis=0)
    e = tl.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    result = (e / denom).to(tl.bfloat16)

    tl.store(out_ptr + row * out_stride + cols, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = y + self.b1
            y = y + self.b2
            return torch.softmax(y, dim=-1)

        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu_bias_softmax[(n_rows,)](
            x, self.b1, self.b2, out,
            n_cols,
            x.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
