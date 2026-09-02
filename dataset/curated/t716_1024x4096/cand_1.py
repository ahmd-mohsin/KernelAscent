import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 716
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact GELU (erf-based), matching F.gelu default
    y = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    tl.store(Y + offs, y.to(tl.float16), mask=mask)


@triton.jit
def _bias_softmax_kernel(X, B, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    # bias add in fp16 semantics first (match x + b2 in fp16), then softmax in fp32
    z = (x.to(tl.float16) + b.to(tl.float16)).to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        n = self.W1.shape[1]

        # Fused exact-GELU (fp32 internal math)
        gx = torch.empty_like(x)
        n_elem = x.numel()
        BLOCK = 1024
        _gelu_kernel[(triton.cdiv(n_elem, BLOCK),)](x, gx, n_elem, BLOCK=BLOCK, num_warps=4)

        # cuBLAS fp16 GEMM (fp32 accumulate)
        y = torch.matmul(gx, self.W1)

        # Fused bias-add + softmax
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_softmax_kernel[(m,)](
            y, self.b2, out, n, y.stride(0), out.stride(0),
            BLOCK=BLOCK_N, num_warps=16,
        )
        return out
