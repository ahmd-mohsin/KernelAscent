import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 288
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_epilogue_kernel(
    X_ptr, B1_ptr, W2_ptr, G4_ptr, B4_ptr, W5_ptr, Out_ptr,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    x = tl.load(X_ptr + row * N + cols).to(tl.float32)

    # bias add (bf16 rounding to match reference)
    b1 = tl.load(B1_ptr + cols).to(tl.float32)
    x = (x + b1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm #2
    ms = tl.sum(x * x, axis=0) / N
    y = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2_ptr + cols).to(tl.float32)
    x = (y * w2).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based), fp32 math with bf16 rounding
    x = (0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (eps=1e-5, biased variance, fp32 accumulation)
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g4 = tl.load(G4_ptr + cols).to(tl.float32)
    b4 = tl.load(B4_ptr + cols).to(tl.float32)
    x = (d * inv * g4 + b4).to(tl.bfloat16).to(tl.float32)

    # RMSNorm #5
    ms2 = tl.sum(x * x, axis=0) / N
    y = (x * tl.math.rsqrt(ms2 + 1e-6)).to(tl.bfloat16).to(tl.float32)
    w5 = tl.load(W5_ptr + cols).to(tl.float32)
    out = (y * w5).to(tl.bfloat16)

    tl.store(Out_ptr + row * N + cols, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 tensor-core matmul
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        rows, N = y.shape
        out = torch.empty_like(y)
        _fused_epilogue_kernel[(rows,)](
            y, self.b1, self.rms2_w, self.ln4_g, self.ln4_b, self.rms5_w, out,
            N=N, BLOCK=N,
            num_warps=8,
        )
        return out
