import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 774
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G1, B1, B2, G4, B4, Out,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, round to bf16 like PyTorch output) ----
    m = tl.max(x, 0)
    e = tl.exp(x - m)                     # masked lanes: exp(-inf) = 0
    s = tl.sum(e, 0)
    sm = e / s
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # ---- layer_norm 1 (fp32 stats, eps=1e-5) ----
    mean1 = tl.sum(tl.where(mask, sm, 0.0), 0) / D
    d1 = tl.where(mask, sm - mean1, 0.0)
    var1 = tl.sum(d1 * d1, 0) / D
    rstd1 = 1.0 / tl.sqrt(var1 + 1e-5)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- add bias (round to bf16) ----
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y + b2).to(tl.bfloat16).to(tl.float32)

    # ---- scale (round to bf16) ----
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # ---- layer_norm 4 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), 0) / D
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + 1e-5)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g4 + b4

    tl.store(Out + row * D + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            # fallback (reference path)
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = x + self.b2
            x = x * 1.2769
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        n_rows = xc.shape[0]
        out = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(n_rows,)](
            xc, self.ln1_g, self.ln1_b, self.b2, self.ln4_g, self.ln4_b, out,
            D=d, SCALE=1.2769, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
