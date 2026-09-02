import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 821
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_ln_softmax_ln(
    X, G1, B1, G2, B2, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (fp32 accumulation, like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g1 + b1
    # round to bf16 (output dtype between ops in reference)
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32) ----
    y_m = tl.where(mask, y, float("-inf"))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean2 = tl.sum(p, axis=0) / N
    d2 = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g2 + b2

    tl.store(Y + row * stride_y + offs, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 tensor-core matmul
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        grid = (Mrows,)
        _fused_ln_softmax_ln[grid](
            h, self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b, out,
            h.stride(0), out.stride(0),
            N=N, EPS=1e-5, BLOCK=512,
            num_warps=8,
        )
        return out
