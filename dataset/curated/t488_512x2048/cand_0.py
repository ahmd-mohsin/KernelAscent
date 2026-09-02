import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 488
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_scale_ln_softmax_ln(
    X, Y, G1, B1, G3, B3,
    D: tl.constexpr, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- scale (compute in fp32, round back to bf16 like eager) ----
    x = x * 1.2406
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 1 ----
    mean = tl.sum(x, axis=0) / D
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g1 + b1
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax ----
    mval = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.where(mask, tl.exp(x - mval), 0.0)
    s = tl.sum(e, axis=0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 ----
    mean2 = tl.sum(x, axis=0) / D
    d2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d2 * rstd2 * g3 + b3

    tl.store(Y + row * D + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x * 1.2406
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.softmax(y, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return y

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_scale_ln_softmax_ln[(n_rows,)](
            x2, y,
            self.ln1_g, self.ln1_b,
            self.ln3_g, self.ln3_b,
            d, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
