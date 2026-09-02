import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 487
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_rms_rms_gelu(X, W1, W2, Y, N, stride, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row * stride + offs

    x = tl.load(ptr, mask=mask, other=0.0)          # fp16
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)   # fp16
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)   # fp16

    # RMSNorm 1 (compute in fp32, cast to fp16, multiply by weight in fp16)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    x = (xf * inv).to(tl.float16) * w1

    # RMSNorm 2
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    x = (xf * inv).to(tl.float16) * w2

    # GELU (erf-based, computed in fp32 like PyTorch opmath for half)
    xf = x.to(tl.float32)
    y = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    y = y.to(tl.float16)

    tl.store(Y + row * stride + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        _fused_rms_rms_gelu[(m,)](
            x, self.rms1_w, self.rms2_w, y,
            n, x.stride(0), 1e-6,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return y
