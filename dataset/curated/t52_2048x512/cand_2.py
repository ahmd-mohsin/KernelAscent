import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 52
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_rms_relu_rms(X, W1, B3, W4, Out, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    ptr = X + row * stride

    xf = tl.load(ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 1
    ms1 = tl.sum(xf * xf, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    y = (xf * r1).to(tl.bfloat16).to(tl.float32)

    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w1).to(tl.bfloat16).to(tl.float32)

    # ReLU
    y = tl.maximum(y, 0.0)

    # + b3
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b3).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    ms2 = tl.sum(tl.where(mask, y * y, 0.0), axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    z = (y * r2).to(tl.bfloat16).to(tl.float32)

    w4 = tl.load(W4 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z * w4).to(tl.bfloat16)

    tl.store(Out + row * stride + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        M_, N_ = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _fused_rms_relu_rms[(M_,)](
            x, self.rms1_w, self.b3, self.rms4_w, out,
            N_, x.stride(0), BLOCK=BLOCK,
            num_warps=8,
        )
        return out
