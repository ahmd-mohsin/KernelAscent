import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 774
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, Y,
    G1, B1, B2, G4, B4,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 math, round to bf16 like PyTorch op boundary) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- layer_norm 1 ----
    n_f = N.to(tl.float32)
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / n_f
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / n_f
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y - mean) * rstd * g1 + b1
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- add bias ----
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = y + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- scale ----
    y = y * SCALE
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- layer_norm 2 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), axis=0) / n_f
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / n_f
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y - mean2) * rstd2 * g4 + b4

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            y = (y + self.b2) * 1.2769
            return F.layer_norm(y, (y.shape[-1],), self.ln4_g, self.ln4_b)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(rows,)](
            x2, out,
            self.ln1_g, self.ln1_b, self.b2, self.ln4_g, self.ln4_b,
            n, x2.stride(0), out.stride(0),
            EPS=1e-5, SCALE=1.2769,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
