import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 834
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _relu_rmsnorm_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    D_: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    # mean of squares over D
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + EPS)

    xn = (xf * inv).to(Y.dtype.element_ty)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w
    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, Dcols = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2 = x.view(-1, Dcols)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)
        w = self.rms1_w
        BLOCK = triton.next_power_of_2(Dcols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_rmsnorm_kernel[(n_rows,)](
            x2, w, y,
            x2.stride(0), y.stride(0),
            Dcols, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view_as(x)
