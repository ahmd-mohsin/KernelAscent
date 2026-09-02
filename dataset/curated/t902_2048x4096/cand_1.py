import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 902
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_softmax2_gelu(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 accumulate, like PyTorch half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = e1 / s1
    # round intermediate to fp16 (matches materialized half tensor between ops)
    y1 = y1.to(tl.float16).to(tl.float32)

    # softmax 2
    y1m = tl.where(mask, y1, float('-inf'))
    m2 = tl.max(y1m, axis=0)
    e2 = tl.exp(y1m - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = e2 / s2
    y2 = y2.to(tl.float16).to(tl.float32)

    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = y2 * 0.5 * (1.0 + tl.math.erf(y2 * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return F.gelu(x)

        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2d.shape
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_softmax2_gelu[(n_rows,)](
            x2d, out, n_cols,
            x2d.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
