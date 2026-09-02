import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 786
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_ln_kernel(
    X, B0, G, B, Y,
    N, scale,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    x = x + b0

    mean = tl.sum(x, axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (xm * rstd * g + b) * scale
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_ln_kernel[(Mrows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, y,
            N, 1.4996,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
