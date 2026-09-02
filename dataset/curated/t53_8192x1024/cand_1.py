import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 53
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _rms_scale_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N, eps,
    s0, s1, s2,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    # normalized value cast back to bf16 (matches .to(x.dtype))
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)

    # bf16 tensor-tensor multiply (single rounding, matches PyTorch bf16 mul)
    y = xn * w

    # each scalar multiply: compute in fp32, round back to bf16 (matches PyTorch)
    y = (y.to(tl.float32) * s0).to(tl.bfloat16)
    y = (y.to(tl.float32) * s1).to(tl.bfloat16)
    y = (y.to(tl.float32) * s2).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda and x.dim() == 2
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        w = self.rms0_w
        if w.device != x.device:
            w = w.to(x.device)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_scale_kernel[(Mrows,)](
            x, w, y,
            x.stride(0), y.stride(0),
            N, 1e-6,
            1.4888, 1.1305, 1.0234,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
