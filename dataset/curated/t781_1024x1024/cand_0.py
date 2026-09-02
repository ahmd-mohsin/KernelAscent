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
    X, LN1G, LN1B, LN3G, LN3B, B4, OUT,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, like PyTorch) ----
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    inv1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g1 = tl.load(LN1G + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(LN1B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * inv1 * g1 + b1
    # round to bf16 (LN output dtype in reference)
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- scale by 1.2185 (fp32 opmath, bf16 result) ----
    y = y * 1.2185
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g3 = tl.load(LN3G + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(LN3B + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * inv2 * g3 + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- add bias (fp32 opmath, bf16 result) ----
    bb = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    z = z + bb
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU: 0.5*z*(1+erf(z/sqrt(2))) in fp32 ----
    out = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(OUT + row * stride + cols, out.to(tl.bfloat16), mask=mask)


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
        # cuBLAS tensor-core matmul
        y = torch.matmul(x, self.W0)
        if not y.is_contiguous():
            y = y.contiguous()

        rows, N = y.shape[0], y.shape[1]
        out = torch.empty_like(y)

        _fused_ln_ln_gelu[(rows,)](
            y, self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b, self.b4, out,
            N, y.stride(0),
            BLOCK=512,
            num_warps=4,
        )
        return out
