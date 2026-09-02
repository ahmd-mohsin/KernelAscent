import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 807
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_bias_rms_relu_rms(
    X, B1, W2, W4, OUT,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + offs)          # fp16
    b1 = tl.load(B1 + offs)                  # fp16
    x = x + b1                               # fp16 add (matches PyTorch)

    # rmsnorm 1 (in fp32)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * inv).to(tl.float16)
    w2 = tl.load(W2 + offs)
    x = xh * w2                              # fp16 mul

    # relu
    zero = tl.zeros_like(x)
    x = tl.maximum(x, zero)

    # rmsnorm 2 (in fp32)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (xf * inv).to(tl.float16)
    w4 = tl.load(W4 + offs)
    x = xh * w4                              # fp16 mul

    tl.store(OUT + row * N + offs, x)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS tensor-core GEMM
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        _fused_bias_rms_relu_rms[(m,)](
            y, self.b1, self.rms2_w, self.rms4_w, out,
            N=n, BLOCK=n,
            num_warps=8,
        )
        return out
