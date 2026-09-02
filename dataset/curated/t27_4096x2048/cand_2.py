import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 27
M, D, DT = 4096, 2048, torch.float16


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
    inv = 1.0 / tl.sqrt(ms + EPS)

    # normalized value cast to fp16 (matches .to(x.dtype))
    n = (x * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16

    # fp16 * fp16 multiply (matches half tensor * half tensor on CUDA)
    y = n * w

    # scalar multiply computed in fp32 opmath then cast back (matches half * python float)
    y = (y.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rmsnorm_scale_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            N=n, EPS=1e-6, SCALE=1.1239,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
