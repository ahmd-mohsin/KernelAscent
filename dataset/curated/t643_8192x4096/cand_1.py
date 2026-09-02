import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 643
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_relu_gelu2_softmax_kernel(
    X_ptr, Y_ptr,
    n_cols,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact erf), round to bf16 to match reference intermediate precision
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    g = g * 0.5 * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch)
    g_masked = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2:
            x = x.contiguous()
            n_rows, n_cols = x.shape
            y = torch.empty_like(x)
            BLOCK = triton.next_power_of_2(n_cols)
            num_warps = 8 if BLOCK >= 2048 else 4
            _fused_relu_gelu2_softmax_kernel[(n_rows,)](
                x, y, n_cols,
                x.stride(0), y.stride(0),
                BLOCK=BLOCK,
                num_warps=num_warps,
            )
            return y

        # fallback
        x = torch.relu(x)
        x = F.gelu(x)
        x = F.gelu(x)
        x = torch.softmax(x, dim=-1)
        return x
