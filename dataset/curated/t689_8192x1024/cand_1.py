import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 689
M, D, DT = 8192, 1024, torch.float16


@triton.jit
def _rms_relu_scale_kernel(X, W, Y, N, scale, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xhat = (x * inv).to(tl.float16)  # cast to fp16 as reference does
    w = tl.load(W + cols, mask=mask, other=0.0)
    # half*half in PyTorch uses fp32 opmath then casts back to half
    y = (xhat.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    # relu in fp16
    y = tl.maximum(y, tl.zeros_like(y))
    # scalar mul: fp32 opmath, cast back to fp16
    y = (y.to(tl.float32) * scale).to(tl.float16)
    tl.store(Y + row * N + cols, y, mask=mask)


@triton.jit
def _rms_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xhat = (x * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xhat.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N1 = x.shape
        y1 = torch.empty_like(x)
        _rms_relu_scale_kernel[(Mrows,)](
            x, self.rms0_w, y1, N1, 1.0454, 1e-6,
            BLOCK=triton.next_power_of_2(N1), num_warps=8
        )
        y2 = y1 @ self.W3
        N2 = y2.shape[1]
        out = torch.empty_like(y2)
        _rms_kernel[(Mrows,)](
            y2, self.rms4_w, out, N2, 1e-6,
            BLOCK=triton.next_power_of_2(N2), num_warps=4
        )
        return out
