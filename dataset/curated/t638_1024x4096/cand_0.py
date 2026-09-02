import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 638
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _fused_softmax_ln_relu(
    X, G, B, Y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load row, cast to fp32
    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # x = x * 1.1576  (fp32 math, round to bf16 like the eager op)
    x = (x * 1.1576).to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation, bf16 output rounding)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # x = x * 1.3589 (round to bf16)
    p = (p * 1.3589).to(tl.bfloat16).to(tl.float32)
    p = tl.where(mask, p, 0.0)

    # layer norm (fp32 stats)
    mean = tl.sum(p, axis=0) / D
    diff = tl.where(mask, p - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * g + b
    # relu
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * D + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = x * 1.1576
            x = torch.softmax(x, dim=-1)
            x = x * 1.3589
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return torch.relu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        rows = xc.shape[0]
        out = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_ln_relu[(rows,)](
            xc, self.ln3_g, self.ln3_b, out,
            D=d, EPS=1e-5, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
