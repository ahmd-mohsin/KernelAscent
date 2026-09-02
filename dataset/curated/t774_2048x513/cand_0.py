import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 774
M, D, DT = 2048, 513, torch.bfloat16


@triton.jit
def _fused_softmax_ln_ln_kernel(
    X, OUT,
    G1, B1, B2P, G4, B4,
    N,
    stride_x, stride_o,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, round to bf16 like PyTorch output) ----
    mx = tl.max(x, 0)
    e = tl.exp(x - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = e / s
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- layer_norm 1 ----
    mean = tl.sum(tl.where(mask, y, 0.0), 0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * g1 + b1).to(tl.bfloat16).to(tl.float32)

    # ---- + b2 (bf16 rounding at op boundary) ----
    b2 = tl.load(B2P + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b2).to(tl.bfloat16).to(tl.float32)

    # ---- * 1.2769 ----
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # ---- layer_norm 4 ----
    mean2 = tl.sum(tl.where(mask, y, 0.0), 0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    g4 = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g4 + b4

    tl.store(OUT + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


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
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_softmax_ln_ln_kernel[(rows,)](
            x2, out,
            self.ln1_g, self.ln1_b, self.b2, self.ln4_g, self.ln4_b,
            N,
            x2.stride(0), out.stride(0),
            EPS=1e-5,
            SCALE=1.2769,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
