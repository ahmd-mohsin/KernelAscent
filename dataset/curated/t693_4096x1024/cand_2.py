import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 693
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _bias_add_kernel(X, B, Y, n_elements, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask)
    b = tl.load(B + (offs % N), mask=mask)
    y = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    tl.store(Y + offs, y, mask=mask)


@triton.jit
def _scale_softmax_kernel(X, Y, N, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    # match reference: scale is applied in bf16, then softmax accumulates in fp32
    x = (x * scale).to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = x @ self.W1
            x = x * 1.2082
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        Mrows, Dcols = x.shape

        # fused bias add (elementwise, single kernel)
        xb = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK),)
        _bias_add_kernel[grid](x, self.b0, xb, n_elements, Dcols, BLOCK=BLOCK)

        # matmul via cuBLAS tensor cores
        h = xb @ self.W1
        h = h.contiguous()

        # fused scale + softmax (one pass per row)
        N = h.shape[1]
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _scale_softmax_kernel[(h.shape[0],)](
            h, out, N, 1.2082, BLOCK=BLOCK_N, num_warps=num_warps
        )
        return out
