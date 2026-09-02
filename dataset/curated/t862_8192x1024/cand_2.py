import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 862
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _ln_scale_gelu_kernel(
    X, G, B, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * g + b
    y = y * SCALE

    # exact GELU: 0.5 * y * (1 + erf(y / sqrt(2)))
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        M_, N_ = h.shape
        y = torch.empty_like(h)
        _ln_scale_gelu_kernel[(M_,)](
            h, self.ln1_g, self.ln1_b, y,
            h.stride(0), y.stride(0),
            N=N_, EPS=1e-5, SCALE=1.4752,
            BLOCK=triton.next_power_of_2(N_),
            num_warps=4,
        )
        return y
