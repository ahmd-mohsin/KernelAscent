import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 991
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _ln_kernel(X, G, B, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean) * rstd * g + b
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


@triton.jit
def _scale_softmax_kernel(X, Y, N, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # replicate bf16 rounding of the elementwise scale before softmax
    x = (x * scale).to(tl.bfloat16).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]

        ln_out = torch.empty_like(x2)
        BLOCK_LN = triton.next_power_of_2(N)
        _ln_kernel[(rows,)](
            x2, self.ln0_g, self.ln0_b, ln_out, N, 1e-5,
            BLOCK=BLOCK_LN, num_warps=8,
        )

        h = ln_out @ self.W1  # cuBLAS bf16 tensor-core GEMM

        K = h.shape[-1]
        out = torch.empty_like(h)
        BLOCK_SM = triton.next_power_of_2(K)
        _scale_softmax_kernel[(rows,)](
            h, out, K, 1.4415,
            BLOCK=BLOCK_SM, num_warps=4,
        )

        return out.view(*orig_shape[:-1], K)
