import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 679
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _double_rmsnorm_kernel(
    X_ptr, W0_ptr, W1_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # First RMSNorm
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w0 = tl.load(W0_ptr + cols, mask=mask, other=0.0)
    # cast to fp16 then multiply in fp16 to match reference exactly
    x1 = (xf * r).to(tl.float16) * w0

    # Second RMSNorm
    x1f = x1.to(tl.float32)
    ms2 = tl.sum(x1f * x1f, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w1 = tl.load(W1_ptr + cols, mask=mask, other=0.0)
    y = (x1f * r2).to(tl.float16) * w1

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda and x.dim() == 2
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _double_rmsnorm_kernel[(m,)](
            x, self.rms0_w, self.rms1_w, y,
            x.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
