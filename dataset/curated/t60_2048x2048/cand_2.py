import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 60
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_relu_bias_rmsnorm(
    X, B1, W, Y,
    stride_xm, stride_ym,
    N, EPS,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)

    # relu + bias (in bf16, matching reference), then compute in fp32
    x = tl.maximum(x, 0.0) + b1
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + EPS)

    xn = (xf * rstd).to(Y.dtype.element_ty)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w
    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _fused_relu_bias_rmsnorm[(Mrows,)](
            x2, self.b1, self.rms2_w, y,
            x2.stride(0), y.stride(0),
            N, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
