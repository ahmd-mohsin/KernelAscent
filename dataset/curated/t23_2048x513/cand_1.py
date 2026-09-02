import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 23
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _gelu_bias_kernel(X, B, Y, n_cols, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact erf-based GELU computed in fp32 (matches PyTorch opmath), rounded to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    col = offs % n_cols
    b = tl.load(B + col, mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _bias_bias_softmax_kernel(X, B4, B5, Y, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(X + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5 + offs, mask=mask, other=0.0).to(tl.float32)
    # replicate the two separate bf16 additions (with intermediate rounding)
    x = (x + b4).to(tl.bfloat16).to(tl.float32)
    x = (x + b5).to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, -float('inf'))
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = (e / s).to(tl.bfloat16)
    tl.store(Y + row * n_cols + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 2048, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, cols = h.shape
        N = rows * cols

        # fused GELU + bias
        h2 = torch.empty_like(h)
        BLOCK = 1024
        grid = (triton.cdiv(N, BLOCK),)
        _gelu_bias_kernel[grid](h, self.b2, h2, cols, N, BLOCK=BLOCK, num_warps=4)

        # GEMM 2 (cuBLAS tensor cores)
        z = torch.matmul(h2, self.W3).contiguous()
        r, c = z.shape

        # fused bias + bias + softmax
        out = torch.empty_like(z)
        BLOCK_C = triton.next_power_of_2(c)
        _bias_bias_softmax_kernel[(r,)](z, self.b4, self.b5, out, c, BLOCK=BLOCK_C, num_warps=8)
        return out
