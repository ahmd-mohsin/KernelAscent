import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 229
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _softmax_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)
    xmax = tl.max(x, axis=0)
    e = tl.exp(x - xmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _bias_rmsnorm_kernel(X, B, W, Y, n_cols, stride_x, stride_y, eps,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    # bias add rounded to bf16 (matches reference bf16 addition)
    v = (x + b).to(tl.bfloat16).to(tl.float32)
    ms = tl.sum(tl.where(mask, v * v, 0.0), axis=0) / n_cols
    inv = 1.0 / tl.sqrt(ms + eps)
    normed = (v * inv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (normed * w).to(tl.bfloat16)
    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        # fused softmax
        sm = torch.empty_like(x)
        BLOCK_D = triton.next_power_of_2(d)
        _softmax_kernel[(m,)](x, sm, d, x.stride(0), sm.stride(0),
                              BLOCK=BLOCK_D, num_warps=8)
        # matmul via cuBLAS (best on A100 tensor cores)
        y = sm @ self.W1
        # fused bias-add + RMSNorm + weight scale
        n = y.shape[1]
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_rmsnorm_kernel[(m,)](y, self.b2, self.rms3_w, out, n,
                                   y.stride(0), out.stride(0), 1e-6,
                                   BLOCK=BLOCK_N, num_warps=8)
        return out
