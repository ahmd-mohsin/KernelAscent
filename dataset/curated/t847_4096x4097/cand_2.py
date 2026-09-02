import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 847
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _softmax_relu_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_ptr = X + row * stride_xm + cols
    x = tl.load(x_ptr, mask=mask, other=float('-inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom
    # relu is a no-op on softmax outputs (all >= 0), kept implicitly

    y_ptr = Y + row * stride_ym + cols
    tl.store(y_ptr, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            return torch.relu(torch.softmax(x, dim=-1))

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        if BLOCK > 65536:
            return torch.relu(torch.softmax(x, dim=-1))

        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _softmax_relu_kernel[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
