import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 671
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_bias_softmax_gelu2_rms_kernel(
    X_ptr, B_ptr, W_ptr, Out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    # load row (bf16) and bias, add, round to bf16 (matches x + b1 in bf16)
    x = tl.load(X_ptr + base + offs).to(tl.float32)
    b = tl.load(B_ptr + offs).to(tl.float32)
    xb = (x + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32, output rounded to bf16
    m = tl.max(xb, axis=0)
    e = tl.math.exp(xb - m)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # gelu (exact, erf-based) applied twice, rounding to bf16 between ops
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = (0.5 * p * (1.0 + tl.math.erf(p * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    g2 = (0.5 * g1 * (1.0 + tl.math.erf(g1 * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(g2 * g2, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y = (g2 * r).to(tl.bfloat16).to(tl.float32)

    # scale by rms weight (bf16 multiply semantics)
    w = tl.load(W_ptr + offs).to(tl.float32)
    out = (y * w).to(tl.bfloat16)
    tl.store(Out_ptr + base + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        _fused_bias_softmax_gelu2_rms_kernel[(rows,)](
            h, self.b1, self.rms5_w, out,
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out
