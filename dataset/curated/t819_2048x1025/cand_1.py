import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 819
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _fused_relu_scale_softmax_kernel(
    X, Y,
    n_cols,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    xf = x.to(tl.float32)

    # relu(relu(x)) == relu(x); scale in fp32 then round to bf16
    # (matches PyTorch: bf16 tensor * python float uses fp32 opmath, casts back to bf16)
    v = tl.maximum(xf, 0.0) * SCALE
    v = tl.where(mask, v, float('-inf'))
    v = v.to(tl.bfloat16).to(tl.float32)  # replicate bf16 rounding of intermediate

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    row_max = tl.max(v, axis=0)
    e = tl.exp(v - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.2572
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]

        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_relu_scale_softmax_kernel[(n_rows,)](
            x2d, y,
            n_cols,
            x2d.stride(0), y.stride(0),
            SCALE=1.2572,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
