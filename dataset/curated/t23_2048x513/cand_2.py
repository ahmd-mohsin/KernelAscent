import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 23
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _gelu_bias_kernel(X, B, Y, n_elements, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    # exact (erf) GELU in fp32, round to bf16 (matches F.gelu on bf16)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    b = tl.load(B + (offs % N), mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _bias2_softmax_kernel(X, B4, B5, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5 + offs, mask=mask, other=0.0).to(tl.float32)
    # match two separate bf16-rounded additions
    x = (x + b4).to(tl.bfloat16).to(tl.float32)
    x = (x + b5).to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, -float('inf'))
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = (e / s).to(tl.bfloat16)
    tl.store(Y + row * N + offs, y, mask=mask)


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
        # matmul 1 (cuBLAS tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        n = h.numel()
        N1 = h.shape[-1]
        BLOCK = 1024
        grid = (triton.cdiv(n, BLOCK),)
        # fused GELU + bias (in-place)
        _gelu_bias_kernel[grid](h, self.b2, h, n, N1, BLOCK=BLOCK)

        # matmul 2 (cuBLAS tensor cores)
        out = (h @ self.W3).contiguous()
        N2 = out.shape[-1]
        rows = out.numel() // N2
        y = torch.empty_like(out)
        # fused bias + bias + softmax
        _bias2_softmax_kernel[(rows,)](out, self.b4, self.b5, y, N2,
                                       BLOCK=triton.next_power_of_2(N2))
        return y
