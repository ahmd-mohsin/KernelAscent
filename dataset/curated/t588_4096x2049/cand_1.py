import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 588
M, D, DT = 4096, 2049, torch.bfloat16


@triton.jit
def _double_rmsnorm_kernel(
    X, W0, W1, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # First RMSNorm
    ms0 = tl.sum(x * x, axis=0) / N
    r0 = tl.math.rsqrt(ms0 + EPS)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    x1 = (x * r0).to(tl.bfloat16) * w0  # bf16 multiply, matches PyTorch semantics

    # Second RMSNorm
    x1f = x1.to(tl.float32)
    ms1 = tl.sum(x1f * x1f, axis=0) / N
    r1 = tl.math.rsqrt(ms1 + EPS)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    y = (x1f * r1).to(tl.bfloat16) * w1

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4

        _double_rmsnorm_kernel[(Mrows,)](
            x2d, self.rms0_w, self.rms1_w, y,
            N, x2d.stride(0), y.stride(0),
            EPS=1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
