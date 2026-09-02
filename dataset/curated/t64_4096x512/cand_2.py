import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 64
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _double_rmsnorm_kernel(
    X, W1, W2, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # First RMSNorm
    ms1 = tl.sum(x * x, axis=0) / N
    inv1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    y = x * inv1
    # round to bf16 (matches .to(x.dtype)) then multiply by weight in bf16 semantics
    y = y.to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w1).to(tl.bfloat16).to(tl.float32)

    # Second RMSNorm
    ms2 = tl.sum(y * y, axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    z = y * inv2
    z = z.to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z * w2).to(tl.bfloat16)

    tl.store(Out + row * stride_o + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        M_, N_ = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _double_rmsnorm_kernel[(M_,)](
            x, self.rms1_w, self.rms2_w, out,
            x.stride(0), out.stride(0),
            N_, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
