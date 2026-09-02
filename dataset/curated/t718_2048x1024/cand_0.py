import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 718
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_ln_gelu_ln(
    X, Y, G0, B0, G2, B2,
    N, eps,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 0 (fp32 math, like PyTorch's mixed-precision LN)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y0 = xc * rstd * g0 + b0
    # cast to bf16 (LN output dtype) then back to fp32 for gelu opmath
    y0 = y0.to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), fp32 opmath, cast to bf16 like PyTorch elementwise kernel
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    ge = y0 * 0.5 * (1.0 + tl.math.erf(y0 * INV_SQRT2))
    ge = ge.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, ge, 0.0), axis=0) / N
    gc = tl.where(mask, ge - mean2, 0.0)
    var2 = tl.sum(gc * gc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = gc * rstd2 * g2 + b2

    tl.store(Y + row * stride_y + cols, y2.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_ln_gelu_ln[(Mrows,)](
            x2d, y,
            self.ln0_g, self.ln0_b, self.ln2_g, self.ln2_b,
            N, 1e-5,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
