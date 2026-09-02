import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 510
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _double_ln_kernel(
    X, Y,
    G0, B0, G1, B1,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, like PyTorch's opmath for bf16) ----
    mean0 = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(xc * xc, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + EPS)

    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y0 = xc * rstd0 * g0 + b0

    # PyTorch casts back to bf16 between the two layer_norm calls
    y0 = y0.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 1 ----
    y0m = tl.where(mask, y0, 0.0)
    mean1 = tl.sum(y0m, axis=0) / N
    yc = tl.where(mask, y0 - mean1, 0.0)
    var1 = tl.sum(yc * yc, axis=0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + EPS)

    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y1 = yc * rstd1 * g1 + b1

    tl.store(Y + row * stride_y + cols, y1.to(tl.bfloat16), mask=mask)


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
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4

        _double_ln_kernel[(m,)](
            x2, y,
            self.ln0_g, self.ln0_b, self.ln1_g, self.ln1_b,
            n, x2.stride(0), y.stride(0),
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
