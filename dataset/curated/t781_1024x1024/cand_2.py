import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 781
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_ln_ln_gelu(
    X, Y, G1, B1, G3, B3, B4,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # --- LayerNorm 1 ---
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g1 + b1
    # round to bf16 (matches PyTorch intermediate output dtype)
    y = y.to(tl.bfloat16).to(tl.float32)

    # --- scale ---
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # --- LayerNorm 2 ---
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    yc = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(yc * yc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z = yc * rstd2 * g3 + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # --- add bias ---
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z + b4).to(tl.bfloat16).to(tl.float32)

    # --- exact GELU ---
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


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
        h = x @ self.W0  # cuBLAS bf16 GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_ln_gelu[(Mrows,)](
            h, out,
            self.ln1_g, self.ln1_b,
            self.ln3_g, self.ln3_b,
            self.b4,
            N, h.stride(0), out.stride(0),
            EPS=1e-5, SCALE=1.2185,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
