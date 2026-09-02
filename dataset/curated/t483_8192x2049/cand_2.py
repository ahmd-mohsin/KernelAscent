import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 483
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    n_cols,
    stride_xm, stride_ym,
    S1: tl.constexpr, S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # replicate: (fp16 * scalar) computed in fp32, rounded back to fp16 (opmath)
    x = x.to(tl.float32)
    x = (x * S1).to(tl.float16).to(tl.float32)
    x = (x * S2).to(tl.float16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = x * 1.1753
            x = x * 1.0586
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1])
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        m, n = x2.shape
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _scale_softmax_kernel[(m,)](
            x2, y, n,
            x2.stride(0), y.stride(0),
            S1=1.1753, S2=1.0586,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
