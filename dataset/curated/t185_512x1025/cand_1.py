import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 185
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_gelu_relu_softmax_gelu2_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # GELU (exact, computed in fp32 like PyTorch opmath, rounded back to bf16)
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ReLU
    r = tl.maximum(g, 0.0)

    # Softmax over the row (fp32 accumulation, bf16 output like PyTorch)
    r_masked = tl.where(mask, r, float('-inf'))
    row_max = tl.max(r_masked, axis=0)
    e = tl.exp(r_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = (e / denom).to(tl.bfloat16).to(tl.float32)

    # GELU
    g2 = 0.5 * sm * (1.0 + tl.math.erf(sm * INV_SQRT2))
    g2 = g2.to(tl.bfloat16).to(tl.float32)

    # GELU
    g3 = 0.5 * g2 * (1.0 + tl.math.erf(g2 * INV_SQRT2))

    tl.store(Y_ptr + row * stride_y + offs, g3.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if (not x.is_cuda) or x.dtype != torch.bfloat16:
            # Fallback: reference path
            x = F.gelu(x)
            x = torch.relu(x)
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            x = F.gelu(x)
            return x

        orig_shape = x.shape
        n = orig_shape[-1]
        x2d = x.contiguous().view(-1, n)
        m = x2d.shape[0]

        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_gelu_relu_softmax_gelu2_kernel[(m,)](
            x2d, y,
            n, x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
