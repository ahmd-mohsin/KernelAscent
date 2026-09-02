import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 223
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _double_ln_kernel(
    X, Y,
    G0, B0, G1, B1,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, like PyTorch) ----
    mean0 = tl.sum(x, axis=0) / N
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + eps)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    h = d0 * rstd0 * g0 + b0

    # Round intermediate to bf16 to match the reference (LN0 output is bf16)
    h = h.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 1 ----
    mean1 = tl.sum(tl.where(mask, h, 0.0), axis=0) / N
    d1 = tl.where(mask, h - mean1, 0.0)
    var1 = tl.sum(d1 * d1, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g1 + b1

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        xf = x.contiguous().view(-1, N)
        Mrows = xf.shape[0]
        y = torch.empty_like(xf)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _double_ln_kernel[(Mrows,)](
            xf, y,
            self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
