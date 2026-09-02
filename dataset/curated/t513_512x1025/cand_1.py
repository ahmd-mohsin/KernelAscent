import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 513
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_row_kernel(
    X, OUT, G, B, B3,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU (exact, computed in fp32, rounded to fp16 like PyTorch) ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g1 = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g1 = g1.to(tl.float16).to(tl.float32)

    # ---- Softmax 1 (fp32 math, fp16 output rounding) ----
    g1m = tl.where(mask, g1, float("-inf"))
    m1 = tl.max(g1m, axis=0)
    e1 = tl.exp(g1m - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    sm1 = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- LayerNorm (fp32 stats, fp16 output rounding) ----
    nf = N.to(tl.float32)
    mean = tl.sum(tl.where(mask, sm1, 0.0), axis=0) / nf
    diff = tl.where(mask, sm1 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / nf
    rstd = 1.0 / tl.sqrt(var + EPS)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (diff * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---- Add bias (fp32 opmath, fp16 rounding) ----
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (ln + b3).to(tl.float16).to(tl.float32)

    # ---- Softmax 2 ----
    ym = tl.where(mask, y, float("-inf"))
    m2 = tl.max(ym, axis=0)
    e2 = tl.exp(ym - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = (e2 / s2).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = torch.softmax(y, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            y = y + self.b3
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        rows, N = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_row_kernel[(rows,)](
            x2, out, self.ln2_g, self.ln2_b, self.b3,
            N, x2.stride(0), out.stride(0),
            EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
