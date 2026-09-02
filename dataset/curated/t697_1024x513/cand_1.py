import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 697
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_ln_kernel(
    X, OUT,
    G1, B1, B2, G3, B3, G4, B4,
    stride_x, stride_o,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, cast result to fp16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd) * g1 + b1
    y16 = y.to(tl.float16)

    # ---- add b2 in fp16 (matches x + b2 in fp16) ----
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    y16 = y16 + b2

    # ---- LayerNorm 3 ----
    x = y16.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd) * g3 + b3
    y16 = y.to(tl.float16)

    # ---- LayerNorm 4 ----
    x = y16.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd) * g4 + b4

    tl.store(OUT + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 matmul (same as reference)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_kernel[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.b2,
            self.ln3_g, self.ln3_b,
            self.ln4_g, self.ln4_b,
            h.stride(0), out.stride(0),
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
