import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 229
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _bias_rmsnorm_kernel(X, B, W, Y, N, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)
    # bias add in bf16 (matching reference rounding), then upcast
    xb = (x + b)
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * r).to(Y.dtype.element_ty)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        M_, D_ = x.shape
        x = x.contiguous()
        sm = torch.empty_like(x)
        BLOCK_D = triton.next_power_of_2(D_)
        _softmax_kernel[(M_,)](x, sm, D_, x.stride(0), sm.stride(0),
                               BLOCK=BLOCK_D, num_warps=8)
        h = sm @ self.W1  # cuBLAS bf16 matmul with fp32 accumulate
        N = h.shape[1]
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _bias_rmsnorm_kernel[(M_,)](h, self.b2, self.rms3_w, out, N,
                                    h.stride(0), out.stride(0), 1e-6,
                                    BLOCK=BLOCK_N, num_warps=8)
        return out
