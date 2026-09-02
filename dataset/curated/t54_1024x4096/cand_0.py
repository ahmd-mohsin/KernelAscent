import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 54
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _ln_scale_kernel(X, G, B, Y, N, eps, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # match reference: LN output cast to bf16, then scaled (fp32 opmath), cast back
    y = y.to(tl.bfloat16)
    y = (y.to(tl.float32) * scale).to(tl.bfloat16)
    tl.store(Y + row * N + cols, y, mask=mask)


@triton.jit
def _softmax_relu_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    # relu is a no-op on softmax output (all >= 0), but apply for exactness
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 via cuBLAS (already optimal on A100 tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape

        # Fused LayerNorm + scale (single pass, fp32 accumulation)
        y = torch.empty_like(h)
        _ln_scale_kernel[(rows,)](
            h, self.ln1_g, self.ln1_b, y,
            N, 1e-5, 1.3619,
            BLOCK=4096, num_warps=8,
        )

        # GEMM 2 via cuBLAS
        z = y @ self.W3
        z = z.contiguous()
        rows2, N2 = z.shape

        # Fused softmax + relu
        out = torch.empty_like(z)
        _softmax_relu_kernel[(rows2,)](
            z, out, N2,
            BLOCK=512, num_warps=4,
        )
        return out
