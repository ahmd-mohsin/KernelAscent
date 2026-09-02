import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 187
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _gelu2_softmax_kernel(
    X, Y,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    ptr = X + row * n_cols + offs
    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # first GELU (erf-based, fp32 compute, round to fp16 to match reference)
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # second GELU
    g = g * 0.5 * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # softmax over the row (fp32 accumulation, like PyTorch on fp16)
    g = tl.where(mask, g, -float('inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y + row * n_cols + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = F.gelu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _gelu2_softmax_kernel[(n_rows,)](
            x2d, out, n_cols,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
