import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 957
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _rmsnorm_kernel(
    X, W, Y,
    stride_x, stride_y,
    N,
    SCALE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    if APPLY_SCALE:
        # replicate fp16 rounding of the scalar multiply in the reference
        x = (x.to(tl.float32) * SCALE).to(tl.float16)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + 1e-6)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xf * rstd).to(tl.float16) * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


def _rmsnorm(x, w, scale=1.0, apply_scale=False):
    Mrows, N = x.shape
    y = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(N)
    _rmsnorm_kernel[(Mrows,)](
        x, w, y,
        x.stride(0), y.stride(0),
        N,
        SCALE=scale,
        APPLY_SCALE=apply_scale,
        BLOCK=BLOCK,
        num_warps=4,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = _rmsnorm(x, self.rms1_w)
        x = x @ self.W2
        x = _rmsnorm(x, self.rms4_w, scale=1.2112, apply_scale=True)
        return x
