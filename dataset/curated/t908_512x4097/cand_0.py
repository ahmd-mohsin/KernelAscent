import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 908
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_ln_gelu_ln_ln(
    X, Y,
    G0, B0, G2, B2, G3, B3,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    n_f = N.to(tl.float32)

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    mean = tl.sum(x, axis=0) / n_f
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)
    g0 = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g0 + b0
    x = x.to(tl.float16).to(tl.float32)  # match fp16 intermediate rounding

    # ---- GELU (exact, erf) ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_f
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g2 + b2
    x = x.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 3 ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / n_f
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)
    g3 = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = d * rstd * g3 + b3

    tl.store(Y + row * stride_y + offs, x.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return y

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_ln_gelu_ln_ln[(rows,)](
            x2, y,
            self.ln0_g, self.ln0_b,
            self.ln2_g, self.ln2_b,
            self.ln3_g, self.ln3_b,
            N, x2.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
