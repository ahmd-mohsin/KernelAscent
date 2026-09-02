import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 447
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _softmax_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    y = e / s
    tl.store(Y + row * N + offs, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _layernorm_kernel(X, G, B, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    tl.store(Y + row * N + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # First GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)  # (M, 4096)
        h = h.contiguous()
        Mrows, N1 = h.shape

        # Fused softmax (in-place, fp32 accumulation)
        _softmax_kernel[(Mrows,)](
            h, h, N1,
            BLOCK=triton.next_power_of_2(N1),
            num_warps=8,
        )

        # Second GEMM via cuBLAS
        y = torch.matmul(h, self.W2)  # (M, 1024)
        y = y.contiguous()
        _, N2 = y.shape

        out = torch.empty_like(y)
        _layernorm_kernel[(Mrows,)](
            y, self.ln3_g, self.ln3_b, out, N2, 1e-5,
            BLOCK=triton.next_power_of_2(N2),
            num_warps=4,
        )
        return out
