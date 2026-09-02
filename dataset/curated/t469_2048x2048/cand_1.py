import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 469
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, B2, G3, B3, G4, B4, Y,
    D: tl.constexpr,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulation, round to bf16 like PyTorch) ----
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- gelu (erf variant, fp32 math, round to bf16) ----
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = g.to(tl.bfloat16).to(tl.float32)

    # ---- add bias (bf16 arithmetic == fp32 add + round to bf16) ----
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b2).to(tl.bfloat16).to(tl.float32)

    # ---- layernorm 1 (fp32 stats, bf16 output) ----
    mean = tl.sum(tl.where(mask, x, 0.0), 0) / D
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / D
    rstd = 1.0 / tl.sqrt(var + eps)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (d * rstd * g3 + b3).to(tl.bfloat16).to(tl.float32)

    # ---- layernorm 2 (fp32 stats, bf16 output) ----
    mean2 = tl.sum(tl.where(mask, x, 0.0), 0) / D
    d2 = tl.where(mask, x - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (d2 * rstd2 * g4 + b4).to(tl.bfloat16)

    tl.store(Y + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            x = x + self.b2
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x = x.contiguous().view(-1, d)
        m = x.shape[0]
        y = torch.empty_like(x)

        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.b2, self.ln3_g, self.ln3_b, self.ln4_g, self.ln4_b, y,
            d, 1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
