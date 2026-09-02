import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 375
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _softmax_scale_gelu_kernel(
    X, Y,
    n_cols,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, as PyTorch does for bf16 inputs)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    denom = tl.sum(e, axis=0)
    p = e / denom

    # match PyTorch: softmax output cast to bf16, then scale, then gelu
    p = p.to(tl.bfloat16).to(tl.float32)
    y = p * SCALE
    y = y.to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))

    tl.store(Y + row * stride_y + offs, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            x = torch.softmax(x, dim=-1)
            x = x * 1.3079
            return F.gelu(x)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]

        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _softmax_scale_gelu_kernel[(n_rows,)](
            x2d, out,
            n_cols,
            x2d.stride(0), out.stride(0),
            SCALE=1.3079,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
