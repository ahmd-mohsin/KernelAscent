import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 302
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _double_rmsnorm_kernel(
    X, W1, W2, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)                 # fp16
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)                 # fp16

    # First RMSNorm
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    y1 = (xf * inv).to(tl.float16) * w1  # fp16 multiply, matches reference

    # Second RMSNorm
    xf2 = y1.to(tl.float32)
    ms2 = tl.sum(xf2 * xf2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(ms2 + EPS)
    y2 = (xf2 * inv2).to(tl.float16) * w2

    tl.store(Out + row * stride_o + cols, y2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS GEMM
        M_, N_ = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _double_rmsnorm_kernel[(M_,)](
            x, self.rms1_w, self.rms2_w, out,
            x.stride(0), out.stride(0),
            N=N_, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
