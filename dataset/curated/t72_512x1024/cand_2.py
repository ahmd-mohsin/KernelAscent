import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 72
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_ln_relu_ln(
    X, Y, G1, B1, G3, B3,
    D: tl.constexpr, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, round to bf16 like PyTorch output) ----
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)  # emulate bf16 intermediate

    # ---- layer_norm 1 (fp32 math, biased var, eps=1e-5) ----
    mean1 = tl.sum(p, 0) / D
    d1 = tl.where(mask, p - mean1, 0.0)
    var1 = tl.sum(d1 * d1, 0) / D
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g1 + b1

    # ---- relu ----
    y = tl.maximum(y, 0.0)
    y = y.to(tl.bfloat16).to(tl.float32)  # emulate bf16 intermediate

    # ---- layer_norm 3 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), 0) / D
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g3 + b3

    tl.store(Y + row * D + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.relu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return y

        orig_shape = x.shape
        d = orig_shape[-1]
        xr = x.contiguous().view(-1, d)
        rows = xr.shape[0]
        out = torch.empty_like(xr)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_softmax_ln_relu_ln[(rows,)](
            xr, out,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            d, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
