import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 146
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_relu_scale_rmsnorm_kernel(
    X, W, Y,
    stride_x, stride_y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu in fp16
    x = tl.maximum(x, 0.0)
    # multiply by scalar in fp16 (round to fp16 to match reference)
    scale = tl.full((1,), 1.4873, dtype=tl.float16)
    x = (x * scale).to(tl.float16)

    xf = x.to(tl.float32)
    mean_sq = tl.sum(xf * xf, axis=0) / D
    rstd = 1.0 / tl.sqrt(mean_sq + 1e-6)

    xn = (xf * rstd).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        Mrows, Dcols = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(Dcols)
        _fused_relu_scale_rmsnorm_kernel[(Mrows,)](
            x2, self.rms2_w, y,
            x2.stride(0), y.stride(0),
            D=Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
