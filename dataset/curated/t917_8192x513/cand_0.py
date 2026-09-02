import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 917
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G, B, B2, OUT,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 accumulation, biased variance)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b

    # exact GELU (erf)
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    # add bias, relu
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b2
    y = tl.maximum(y, 0.0)

    # softmax
    y = tl.where(mask, y, float('-inf'))
    ymax = tl.max(y, axis=0)
    e = tl.exp(y - ymax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT + row * stride_o + cols, out.to(OUT.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(Mrows,)](
            x, self.ln0_g, self.ln0_b, self.b2, out,
            N, x.stride(0), out.stride(0),
            1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
