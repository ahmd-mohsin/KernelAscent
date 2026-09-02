import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 604
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_rms_kernel(X, W, Y, stride_x, stride_y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)
    # rms
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y = x * inv
    # round to bf16 (matches .to(x.dtype))
    y = y.to(tl.bfloat16).to(tl.float32)
    # * weight (bf16 mul -> round to bf16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)
    # * scalar (bf16 result)
    y = (y * 1.2572).to(tl.bfloat16).to(tl.float32)
    # relu
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
