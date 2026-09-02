import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 97
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _bias_softmax_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)

    # bias add in bf16 (matches reference: x + b0 stored in bf16)
    t_bf16 = (x + b).to(tl.bfloat16)
    t = t_bf16.to(tl.float32)
    t = tl.where(mask, t, float('-inf'))

    m = tl.max(t, axis=0)
    e = tl.exp(t - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_ym + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _bias_softmax_kernel[(Mrows,)](
            x, self.b0, y,
            x.stride(0), y.stride(0),
            Dcols, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
