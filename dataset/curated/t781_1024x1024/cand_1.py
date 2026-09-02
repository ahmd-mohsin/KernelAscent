import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 781
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_ln_ln_gelu_kernel(
    X_ptr, G1_ptr, B1_ptr, G3_ptr, B3_ptr, B4_ptr, Out_ptr,
    stride_x, stride_o,
    N: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, output rounded to bf16 like PyTorch) ----
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)

    g1 = tl.load(G1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean1) * rstd1 * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- scalar multiply (rounded to bf16) ----
    y = (y * 1.2185).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)

    g3 = tl.load(G3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (y - mean2) * rstd2 * g3 + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- bias add (rounded to bf16) ----
    b4 = tl.load(B4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z + b4).to(tl.bfloat16).to(tl.float32)

    # ---- exact (erf) GELU in fp32, then round to bf16 ----
    out = z * 0.5 * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Out_ptr + row * stride_o + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_ln_gelu_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b, self.b4, out,
            h.stride(0), out.stride(0),
            N=N, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
