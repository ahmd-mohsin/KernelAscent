import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 762
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_gelu_bias_ln(X, B2, B3, G, B, Y, N, eps,
                        BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    off = row * N + cols

    x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
    # exact GELU (erf-based), computed in fp32 then cast to fp16 (matches PyTorch)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)

    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    # sequential fp16 adds to match reference rounding
    t16 = (g16 + b2) + b3
    t = t16.to(tl.float32)

    # layernorm in fp32 (matches PyTorch half layernorm)
    mean = tl.sum(t, axis=0) / N
    d = tl.where(mask, t - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * w + bb
    tl.store(Y + off, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS fp16 tensor-core GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_bias_ln[(Mrows,)](
            h, self.b2, self.b3, self.ln4_g, self.ln4_b, y,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
