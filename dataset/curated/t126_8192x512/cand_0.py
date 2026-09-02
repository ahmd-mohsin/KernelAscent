import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 126
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, Y, N, D, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, erf-based) with bf16 rounding to match reference dtype path
    g1 = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g1 = g1.to(tl.bfloat16).to(tl.float32)

    # relu
    r = tl.maximum(g1, 0.0)

    # gelu again
    g2 = 0.5 * r * (1.0 + tl.math.erf(r * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # softmax over the row (fp32 accumulation, like PyTorch)
    g2 = tl.where(mask, g2, float('-inf'))
    m = tl.max(g2, axis=0)
    e = tl.exp(g2 - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = torch.relu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, d = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(n_rows,)](
            x2, y, n_rows, d,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
