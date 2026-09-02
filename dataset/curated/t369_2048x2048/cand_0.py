import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 369
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _rmsnorm_scale_kernel(
    X, W, Y,
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

    ms = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS)

    # match reference rounding: cast normalized value to fp16 first
    y_h = (x * rrms).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    z_h = (y_h.to(tl.float32) * w).to(tl.float16)
    out = (z_h.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rmsnorm_scale_kernel[(rows,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            N=N, EPS=1e-6, SCALE=1.4399,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
