import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 958
M, D, DT = 2048, 4097, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X, OUT, G, B, B4,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, round to bf16 like eager op boundary) ----
    row_max = tl.max(x, axis=0)
    ex = tl.exp(x - row_max)
    ex = tl.where(mask, ex, 0.0)
    denom = tl.sum(ex, axis=0)
    sm = ex / denom
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # ---- exact GELU ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = sm * 0.5 * (1.0 + tl.erf(sm * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)
    g = tl.where(mask, g, 0.0)

    # ---- layernorm (fp32 stats) ----
    mean = tl.sum(g, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bb = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + bb
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- scale ----
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # ---- bias add ----
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b4).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return y * 1.2578 + self.b4

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_row_kernel[(rows,)](
            x2, out, self.ln2_g, self.ln2_b, self.b4,
            N, x2.stride(0), out.stride(0),
            EPS=1e-5, SCALE=1.2578, BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
