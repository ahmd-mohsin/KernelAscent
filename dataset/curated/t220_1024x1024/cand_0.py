import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 220
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_epilogue_kernel(
    X_ptr, B3_ptr, G_ptr, B_ptr, B5_ptr, Out_ptr,
    N, stride_row,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    # exact gelu (erf-based), computed in fp32, rounded to bf16 like PyTorch
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # + b3 (fp32 compute, bf16 round)
    b3 = tl.load(B3_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g + b3).to(tl.bfloat16).to(tl.float32)

    # layernorm in fp32
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = ((y - mean) * inv * gamma + beta).to(tl.bfloat16).to(tl.float32)

    # + b5
    b5 = tl.load(B5_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z + b5).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM with fp32 accumulation
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_epilogue_kernel[(Mrows,)](
            h, self.b3, self.ln4_g, self.ln4_b, self.b5, out,
            N, h.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
