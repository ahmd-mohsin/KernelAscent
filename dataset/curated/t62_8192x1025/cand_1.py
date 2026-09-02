import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 62
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _relu_softmax_scale_kernel(
    X_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)
    # softmax in fp32 (matches PyTorch half softmax which accumulates in float)
    x_m = tl.where(mask, x, float('-inf'))
    m = tl.max(x_m, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    # cast to fp16 (softmax output), then scale in fp32 (opmath) and cast back
    p16 = p.to(tl.float16)
    out = (p16.to(tl.float32) * SCALE).to(tl.float16)
    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


@triton.jit
def _rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    y16 = (x * inv).to(tl.float16)  # (_xf * rsqrt).to(half)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # half * half elementwise -> computed in fp32 opmath, cast to half
    out = (y16.to(tl.float32) * w).to(tl.float16)
    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N1 = h.shape

        # Fused relu + softmax + scale (in-place)
        BLOCK1 = triton.next_power_of_2(N1)
        _relu_softmax_scale_kernel[(Mrows,)](
            h, h,
            h.stride(0), h.stride(0),
            N1,
            SCALE=1.4325,
            BLOCK=BLOCK1,
            num_warps=8,
        )

        # GEMM 2
        y = torch.matmul(h, self.W4)
        y = y.contiguous()
        Mrows2, N2 = y.shape

        # Fused RMSNorm (in-place)
        BLOCK2 = triton.next_power_of_2(N2)
        _rmsnorm_kernel[(Mrows2,)](
            y, self.rms5_w, y,
            y.stride(0), y.stride(0),
            N2,
            EPS=1e-6,
            BLOCK=BLOCK2,
            num_warps=4,
        )
        return y
