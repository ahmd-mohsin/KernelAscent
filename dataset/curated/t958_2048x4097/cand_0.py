import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 958
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_softmax_gelu_ln_kernel(
    X, G, B, B4, Out,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # ---- softmax (fp32 accumulation, output rounded to bf16 like PyTorch) ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = e / denom
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- exact (erf-based) GELU, rounded to bf16 ----
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    g = tl.where(mask, g, 0.0)

    # ---- layer norm (fp32 stats) ----
    mean = tl.sum(g, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- scale (scalar op computed in fp32, rounded to bf16) ----
    y = y * SCALE
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- bias add (opmath fp32, rounded to bf16) ----
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b4

    tl.store(Out + row * stride_o + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        M_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16

        _fused_softmax_gelu_ln_kernel[(M_rows,)](
            x2d, self.ln2_g, self.ln2_b, self.b4, out,
            N, x2d.stride(0), out.stride(0),
            EPS=1e-5,
            SCALE=1.2578,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
