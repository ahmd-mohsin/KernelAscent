import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 677
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, G, B, B3, Y, stride_x, stride_y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based), rounded to bf16 to match PyTorch op boundary
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 accumulation)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g - mean) * rstd * gamma + beta
    y = y.to(tl.bfloat16).to(tl.float32)

    # bias add + relu
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b3
    y = y.to(tl.bfloat16).to(tl.float32)
    y = tl.maximum(y, 0.0)

    # softmax (fp32)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        _fused_kernel[(Mrows,)](
            h, self.ln2_g, self.ln2_b, self.b3, y,
            h.stride(0), y.stride(0),
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return y
