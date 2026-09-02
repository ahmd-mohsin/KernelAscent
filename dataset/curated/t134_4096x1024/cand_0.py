import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 134
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _gelu2_softmax_kernel(
    X, Y,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N_COLS
    ptr = row * N_COLS + offs

    x = tl.load(X + ptr, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # First GELU (exact, erf), round to fp16 to match reference intermediate
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # Second GELU
    g = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax over the row (fp32 accumulation, like PyTorch's half softmax)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + ptr, y.to(tl.float16), mask=mask)


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

        x = x.contiguous()
        rows, cols = x.shape[0] * (x.numel() // (x.shape[-1] * x.shape[0])) if x.dim() > 2 else x.shape[0], x.shape[-1]
        x2d = x.view(-1, cols)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _gelu2_softmax_kernel[(rows,)](
            x2d, out,
            N_COLS=cols,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view_as(x)
