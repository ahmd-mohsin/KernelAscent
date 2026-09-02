import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 167
M, D, DT = 1024, 512, torch.bfloat16

INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_gelu2_softmax(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu #1 (exact, erf-based), round back to bf16 like PyTorch does
    g1 = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g1 = g1.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    g2 = g1 * 0.5 * (1.0 + tl.math.erf(g1 * 0.7071067811865476))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch on bf16 input)
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
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2d.shape
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_gelu2_softmax[(n_rows,)](
            x2d, y, n_cols,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
