import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 817
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_relu_scale_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu in fp16
    x = tl.maximum(x, 0.0)
    # multiply: compute in fp32 (opmath), cast back to fp16 (matches PyTorch)
    xf = x.to(tl.float32) * 1.0204
    xh = xf.to(tl.float16)
    # second relu (no-op numerically but keep for equivalence)
    xh = tl.maximum(xh, 0.0)

    # softmax: upcast to fp32 like PyTorch's fp16 softmax
    v = xh.to(tl.float32)
    v = tl.where(mask, v, float('-inf'))
    row_max = tl.max(v, axis=0)
    e = tl.exp(v - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = torch.relu(x)
            x = x * 1.0204
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _fused_relu_scale_softmax_kernel[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            N=n, BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
