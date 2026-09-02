import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 532
M, D, DT = 1024, 1025, torch.bfloat16


@triton.jit
def _ln_kernel(X, Y, G, B, N, stride_x, stride_y, eps, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _relu_scale_kernel(X, n_elements, scale, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask).to(tl.float32)
    x = tl.maximum(x, 0.0) * scale
    tl.store(X + offs, x.to(X.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        _ln_kernel[(m,)](
            x, y, self.ln0_g, self.ln0_b, n,
            x.stride(0), y.stride(0), 1e-5,
            BLOCK_N=BLOCK_N, num_warps=8,
        )
        out = y @ self.W1
        n_el = out.numel()
        BLOCK = 1024
        _relu_scale_kernel[(triton.cdiv(n_el, BLOCK),)](
            out, n_el, 1.2948, BLOCK=BLOCK, num_warps=4,
        )
        return out
