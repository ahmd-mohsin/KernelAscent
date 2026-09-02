import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 151
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_scale_ln_ln(
    X, Y, G2, B2, G3, B3,
    D: tl.constexpr, scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # --- softmax (fp32 accumulate, round to bf16 like PyTorch) ---
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # --- scale ---
    p = (p * scale).to(tl.bfloat16).to(tl.float32)

    # --- layer norm 2 ---
    mean = tl.sum(p, 0) / D
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, 0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # --- layer norm 3 ---
    mean2 = tl.sum(tl.where(mask, y, 0.0), 0) / D
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g3 + b3

    tl.store(Y + row * D + offs, z.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = x * 1.4033
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_softmax_scale_ln_ln[(rows,)](
            x2, out,
            self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b,
            d, 1.4033, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
