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
    N, stride_x, stride_y,
    scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.2406 (bf16 op: fp32 math, round back to bf16)
    x = (x * scale).to(tl.bfloat16).to(tl.float32)

    nf = N.to(tl.float32)

    # ---- LayerNorm 1 (fp32 math, bf16 output) ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / nf
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / nf
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g1 + b1).to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 math, bf16 output) ----
    y_m = tl.where(mask, y, float("-inf"))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 (fp32 math, bf16 output) ----
    mean2 = tl.sum(tl.where(mask, p, 0.0), axis=0) / nf
    d2 = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / nf
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g3 + b3

    tl.store(Y + row * stride_y + offs, out.to(tl.bfloat16), mask=mask)


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
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_scale_ln_softmax_ln[(m,)](
            x2, out,
            self.ln1_g, self.ln1_b,
            self.ln3_g, self.ln3_b,
            n, x2.stride(0), out.stride(0),
            1.2406, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
