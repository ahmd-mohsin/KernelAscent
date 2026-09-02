import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 656
M, D, DT = 4096, 513, torch.bfloat16


@triton.jit
def _fused_scale_softmax_kernel(
    X_ptr, Y_ptr,
    N,
    stride_row,
    S1: tl.constexpr, S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=float('-inf'))

    # Replicate: (bf16 x * 1.3108) rounded to bf16, then * 1.4024 rounded to bf16
    x = x.to(tl.float32) * S1
    x = x.to(tl.bfloat16).to(tl.float32) * S2
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))

    # Softmax with float32 accumulation (matches PyTorch acc_type for bf16)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_row + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            # Fallback (numerically identical reference path)
            x = x * 1.3108
            x = x * 1.4024
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_scale_softmax_kernel[(rows,)](
            x2, out,
            n,
            x2.stride(0),
            1.3108, 1.4024,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
