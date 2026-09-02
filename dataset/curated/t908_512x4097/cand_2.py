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
    N, stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x_ptr = X + row * stride + cols

    x = tl.load(x_ptr, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g0 + b0
    y = y.to(tl.float16).to(tl.float32)  # match fp16 intermediate rounding

    # ---- GELU (exact, erf-based) ----
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    y = y.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 ----
    y_m = tl.where(mask, y, 0.0)
    mean2 = tl.sum(y_m, axis=0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = d2 * rstd2 * g2 + b2
    z = z.to(tl.float16).to(tl.float32)

    # ---- LayerNorm 3 ----
    z_m = tl.where(mask, z, 0.0)
    mean3 = tl.sum(z_m, axis=0) / N
    d3 = tl.where(mask, z - mean3, 0.0)
    var3 = tl.sum(d3 * d3, axis=0) / N
    rstd3 = 1.0 / tl.sqrt(var3 + EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d3 * rstd3 * g3 + b3

    tl.store(Y + row * stride + cols, out.to(tl.float16), mask=mask)


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
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return x

        x = x.contiguous()
        rows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_ln_gelu_ln_ln[(rows,)](
            x, y,
            self.ln0_g, self.ln0_b,
            self.ln2_g, self.ln2_b,
            self.ln3_g, self.ln3_b,
            N, x.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y
