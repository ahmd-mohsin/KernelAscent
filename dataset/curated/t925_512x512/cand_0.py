import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 925
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _scale_bias_gelu_kernel(Y, B, N, C, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    y = tl.load(Y + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + (offs % C), mask=mask, other=0.0).to(tl.float32)
    y = y * 1.045 + b
    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    tl.store(Y + offs, g.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _softmax_kernel(X, Out, ncols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < ncols
    x = tl.load(X + row * ncols + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Out + row * ncols + offs, y.to(Out.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 via cuBLAS (tensor cores on A100)
        y = (x @ self.W0).contiguous()

        # Fused scale + bias + exact GELU (single memory pass, fp32 math)
        N = y.numel()
        C = y.shape[-1]
        BLOCK = 2048
        grid = (triton.cdiv(N, BLOCK),)
        _scale_bias_gelu_kernel[grid](y, self.b2, N, C, BLOCK=BLOCK, num_warps=8)

        # GEMM 2 via cuBLAS
        z = (y @ self.W4).contiguous()

        # Fused row-wise softmax (one program per row, fp32 accumulation)
        ncols = z.shape[-1]
        rows = z.numel() // ncols
        out = torch.empty_like(z)
        SBLOCK = triton.next_power_of_2(ncols)
        _softmax_kernel[(rows,)](z, out, ncols, BLOCK=SBLOCK, num_warps=8)
        return out
