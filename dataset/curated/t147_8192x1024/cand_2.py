import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 147
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_gelu2_rms_relu(
    X, W, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulate, output rounded to bf16 like PyTorch)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # gelu #1 (exact erf variant, computed in fp32, rounded to bf16)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * p * (1.0 + tl.math.erf(p * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    g = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / D
    r = g * (1.0 / tl.sqrt(ms + 1e-6))
    r = r.to(tl.bfloat16).to(tl.float32)

    # scale by weight (fp32 opmath, cast back to bf16) then relu
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (r * w).to(tl.bfloat16)
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x = x.contiguous().view(-1, d)
        n_rows = x.shape[0]
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_softmax_gelu2_rms_relu[(n_rows,)](
            x, self.rms3_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
