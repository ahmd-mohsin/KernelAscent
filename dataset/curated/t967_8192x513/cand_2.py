import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 967
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _scale_ln_kernel(
    X, Y, G, B,
    N, stride_x, stride_y,
    scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    x = x * scale

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, N = x.shape[0], x.shape[-1]
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _scale_ln_kernel[(rows,)](
            x, y, self.ln1_g, self.ln1_b,
            N, x.stride(0), y.stride(0),
            1.4628, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
