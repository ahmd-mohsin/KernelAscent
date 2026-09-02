import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 656
M, D, DT = 4096, 513, torch.bfloat16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))

    # replicate bf16 rounding of the two scalar multiplies
    x1 = (x.to(tl.float32) * S1).to(tl.bfloat16)
    x2 = (x1.to(tl.float32) * S2).to(tl.bfloat16)

    xf = x2.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
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
            x = x * 1.3108
            x = x * 1.4024
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1]).contiguous()
        m, n = x2d.shape
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _scale_softmax_kernel[(m,)](
            x2d, y, n,
            x2d.stride(0), y.stride(0),
            1.3108, 1.4024,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
