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
    X, Y, N, stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # replicate bf16 rounding of the two multiplies
    x = (x.to(tl.float32) * 1.3108).to(tl.bfloat16)
    x = (x.to(tl.float32) * 1.4024).to(tl.bfloat16)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


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
        x = x.contiguous()
        m, n = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2 = x.view(-1, n)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _scale_softmax_kernel[(rows,)](
            x2, y, n, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view_as(x)
