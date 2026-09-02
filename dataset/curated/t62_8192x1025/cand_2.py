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
    X, Y,
    N,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    # softmax output is rounded to fp16 first (matches PyTorch half softmax),
    # then scaled in fp32 and cast back (matches PyTorch half*scalar semantics)
    p = (e / s).to(tl.float16)
    out = (p.to(tl.float32) * SCALE).to(tl.float16)
    tl.store(Y + row * stride_y + offs, out, mask=mask)


@triton.jit
def _rmsnorm_kernel(
    X, W, Y,
    N,
    stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)
    y16 = (x * r).to(tl.float16)  # cast to half first (matches .to(x.dtype))
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (y16.to(tl.float32) * w).to(tl.float16)  # half*half via fp32 opmath
    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0  # (M, 2048), fp16
        m, n = h.shape

        # Fused relu + softmax + scale (single pass over the row)
        h2 = torch.empty_like(h)
        _relu_softmax_scale_kernel[(m,)](
            h, h2,
            n,
            h.stride(0), h2.stride(0),
            SCALE=1.4325,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )

        # GEMM 2 (cuBLAS tensor cores)
        y = h2 @ self.W4  # (M, 512), fp16
        m2, n2 = y.shape

        # Fused RMSNorm
        out = torch.empty_like(y)
        _rmsnorm_kernel[(m2,)](
            y, self.rms5_w, out,
            n2,
            y.stride(0), out.stride(0),
            EPS=1e-6,
            BLOCK=triton.next_power_of_2(n2),
            num_warps=4,
        )
        return out
