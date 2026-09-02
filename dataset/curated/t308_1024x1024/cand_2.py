import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 308
M, D, DT = 1024, 1024, torch.bfloat16


@triton.jit
def _fused_gelu_ln_softmax_kernel(
    X, Y, G, B,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based) computed in fp32, rounded to bf16 like torch
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 statistics, like torch's bf16 layer_norm)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / D
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)

    w = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * w + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulation, like torch's bf16 softmax)
    ymax = tl.max(tl.where(mask, y, float("-inf")), axis=0)
    e = tl.exp(y - ymax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    out = out.to(tl.bfloat16).to(tl.float32)

    # scale in fp32 opmath, then round to bf16 (matches torch bf16 mul)
    out = out * SCALE
    tl.store(Y + row * D + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = torch.softmax(y, dim=-1)
            return y * 1.0855

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_gelu_ln_softmax_kernel[(n_rows,)](
            x2, out, self.ln1_g, self.ln1_b,
            D=d, EPS=1e-5, SCALE=1.0855, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
