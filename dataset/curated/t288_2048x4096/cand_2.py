import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 288
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X_ptr, B1_ptr, W2_ptr, G4_ptr, B4_ptr, W5_ptr, Out_ptr,
    N,  # row length
    stride_x,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b1  (compute fp32, round to bf16 like PyTorch elementwise op)
    x1 = (x + b1).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    ms = tl.sum(x1 * x1, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = (x1 * rstd).to(tl.bfloat16).to(tl.float32)
    x2 = (x2 * w2).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), fp32 math, round to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x3 = (0.5 * x2 * (1.0 + tl.math.erf(x2 * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # LayerNorm
    mean = tl.sum(tl.where(mask, x3, 0.0), axis=0) / N
    diff = tl.where(mask, x3 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd_ln = 1.0 / tl.sqrt(var + 1e-5)
    g4 = tl.load(G4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x4 = ((x3 - mean) * rstd_ln * g4 + b4).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 5
    ms2 = tl.sum(x4 * x4, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w5 = tl.load(W5_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x5 = (x4 * rstd2).to(tl.bfloat16).to(tl.float32)
    x5 = (x5 * w5).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_x + cols, x5, mask=mask)


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
        # GEMM via cuBLAS (tensor cores)
        y = torch.matmul(x, self.W0)
        out = torch.empty_like(y)
        Mrows, N = y.shape
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            y, self.b1, self.rms2_w, self.ln4_g, self.ln4_b, self.rms5_w, out,
            N, y.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
