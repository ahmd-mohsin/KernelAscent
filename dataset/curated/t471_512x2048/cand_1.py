import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 471
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_gelu_ln_softmax_ln(
    X, G2, B2, G4, B4, Y,
    stride_x, stride_y,
    N,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), then round to bf16 like the eager op does
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 1 (stats in fp32, biased var, eps=1e-5)
    mean1 = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    d1 = tl.where(mask, g - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    inv1 = 1.0 / tl.sqrt(var1 + EPS)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    h = d1 * inv1 * g2 + b2
    h = h.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulate)
    h_m = tl.where(mask, h, float("-inf"))
    mx = tl.max(h_m, axis=0)
    e = tl.exp(h_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, p, 0.0), axis=0) / N
    d2 = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    inv2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d2 * inv2 * g4 + b4

    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores)
        h = x @ self.W0

        orig_shape = h.shape
        N = orig_shape[-1]
        h2 = h.contiguous().view(-1, N)
        rows = h2.shape[0]
        out = torch.empty_like(h2)

        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_softmax_ln[(rows,)](
            h2, self.ln2_g, self.ln2_b, self.ln4_g, self.ln4_b, out,
            h2.stride(0), out.stride(0),
            N,
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
