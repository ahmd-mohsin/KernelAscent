import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 951
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, G1, B1, G2, B2, B3, Y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU: 0.5*x*(1+erf(x/sqrt(2)))
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    # LayerNorm 1
    mean1 = tl.sum(g, axis=0) / N
    d1 = tl.where(mask, g - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = d1 * rstd1 * g1 + b1

    # LayerNorm 2
    mean2 = tl.sum(tl.where(mask, y1, 0.0), axis=0) / N
    d2 = tl.where(mask, y1 - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y2 = d2 * rstd2 * g2 + b2

    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = y2 + b3

    tl.store(Y + row * N + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = F.layer_norm(y, (y.shape[-1],), self.ln2_g, self.ln2_b)
            return y + self.b3

        x = x.contiguous()
        Mrows, N = x.shape[0] if x.dim() == 2 else x.numel() // x.shape[-1], x.shape[-1]
        x2 = x.view(-1, N)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(x2.shape[0],)](
            x2, self.ln1_g, self.ln1_b, self.ln2_g, self.ln2_b, self.b3, y,
            N, 1e-5, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view_as(x)
