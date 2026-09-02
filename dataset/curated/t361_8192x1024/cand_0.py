import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 361
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_ln_sm_ln_act_kernel(
    X, OUT, G1, B1, G3, B3,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, output cast to bf16 like reference) ----
    mean1 = tl.sum(x, axis=0) / N
    d1 = tl.where(mask, x - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x - mean1) * rstd1 * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 math, output cast to bf16) ----
    y_m = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 ----
    mean3 = tl.sum(p, axis=0) / N
    d3 = tl.where(mask, p - mean3, 0.0)
    var3 = tl.sum(d3 * d3, axis=0) / N
    rstd3 = 1.0 / tl.sqrt(var3 + 1e-5)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    z = (p - mean3) * rstd3 * g3 + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- ReLU (cast to bf16 between ops like reference) ----
    z = tl.maximum(z, 0.0)
    z = z.to(tl.bfloat16).to(tl.float32)

    # ---- GELU (exact, erf-based, fp32 math) ----
    out = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(OUT + row * stride_o + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        grid = (Mrows,)
        _fused_ln_sm_ln_act_kernel[grid](
            h, out,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            h.stride(0), out.stride(0),
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=4,
        )
        return out
