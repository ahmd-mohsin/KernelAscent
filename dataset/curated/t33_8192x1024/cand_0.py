import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 33
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_soft_ln_soft_relu_ln(
    X, Y, G1, B1, G4, B4,
    D: tl.constexpr, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 accum, round to bf16 like eager) ----
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    x = e / tl.sum(e, 0)
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- layernorm 1 ----
    mean = tl.sum(tl.where(mask, x, 0.0), 0) / D
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g1 + b1
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- softmax 2 ----
    m2 = tl.max(tl.where(mask, x, float('-inf')), 0)
    e2 = tl.exp(x - m2)
    e2 = tl.where(mask, e2, 0.0)
    x = e2 / tl.sum(e2, 0)
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- relu (no-op after softmax, kept for exactness) ----
    x = tl.maximum(x, 0.0)

    # ---- layernorm 2 ----
    mean2 = tl.sum(tl.where(mask, x, 0.0), 0) / D
    d2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d2 * rstd2 * g4 + b4

    tl.store(Y + row * D + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.softmax(y, dim=-1)
            y = torch.relu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)
            return y

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_soft_ln_soft_relu_ln[(n_rows,)](
            x2, y,
            self.ln1_g, self.ln1_b, self.ln4_g, self.ln4_b,
            d, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
