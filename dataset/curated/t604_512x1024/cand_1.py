import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 604
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_relu_rms_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # relu in input dtype
    x = tl.maximum(x, 0.0)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    # (xf * rsqrt).to(bf16)
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 -> bf16 (computed in fp32, rounded)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    # * scalar -> bf16
    y = (y.to(tl.float32) * SCALE).to(tl.bfloat16)
    # final relu
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = x * 1.2572
            return torch.relu(x)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_relu_rms_kernel[(m,)](
            x2, self.rms1_w, y,
            x2.stride(0), y.stride(0),
            n,
            EPS=1e-6,
            SCALE=1.2572,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y.view(orig_shape)
