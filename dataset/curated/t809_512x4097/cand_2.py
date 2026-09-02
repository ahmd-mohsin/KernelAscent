import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 809
M, D, DT = 512, 4097, torch.bfloat16


@triton.jit
def _fused_act_ln_kernel(
    X_ptr, Out_ptr,
    G4_ptr, B4_ptr, G5_ptr, B5_ptr,
    N, stride_row,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # relu (rounds to bf16 like reference intermediate)
    x = tl.maximum(x, 0.0)
    x = x.to(tl.bfloat16).to(tl.float32)

    # gelu (erf-based), round to bf16 between ops to match reference
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # layernorm 1
    n_f = N * 1.0
    mean1 = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_f
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / n_f
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)
    g4 = tl.load(G4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g4 + b4
    y = y.to(tl.bfloat16).to(tl.float32)

    # layernorm 2
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / n_f
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n_f
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g5 = tl.load(G5_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g5 + b5

    tl.store(Out_ptr + row * stride_row + cols, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores) — identical to reference
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_act_ln_kernel[(m,)](
            h, out,
            self.ln4_g, self.ln4_b, self.ln5_g, self.ln5_b,
            n, h.stride(0),
            EPS=1e-5,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
