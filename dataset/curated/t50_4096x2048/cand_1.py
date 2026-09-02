import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 50
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_softmax_ln_ln_kernel(
    X, Y, G1, B1, G3, B3,
    N, D,
    scale1, scale2, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x_ptr = X + row * D + offs

    # ---- softmax (fp32 accumulation, output rounded to fp16) ----
    x = tl.load(x_ptr, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    p = (e / denom).to(tl.float16).to(tl.float32)
    p = tl.where(mask, p, 0.0)

    # ---- layer_norm 1 (fp32 stats, fp16 output) ----
    mean1 = tl.sum(p, axis=0) / D
    d1 = tl.where(mask, p - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / D
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((p - mean1) * rstd1 * g1 + b1).to(tl.float16)

    # ---- scale 1 (fp32 opmath, round to fp16) ----
    y = (y.to(tl.float32) * scale1).to(tl.float16).to(tl.float32)
    y = tl.where(mask, y, 0.0)

    # ---- layer_norm 2 (fp32 stats, fp16 output) ----
    mean2 = tl.sum(y, axis=0) / D
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / D
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    z = ((y - mean2) * rstd2 * g3 + b3).to(tl.float16)

    # ---- scale 2 (fp32 opmath, round to fp16) ----
    z = (z.to(tl.float32) * scale2).to(tl.float16)

    tl.store(Y + row * D + offs, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_softmax_ln_ln_kernel[(n,)](
            x2, y,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            n, d,
            1.1736, 1.0344, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
