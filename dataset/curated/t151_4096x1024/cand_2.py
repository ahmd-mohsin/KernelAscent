import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 151
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_double_ln(
    X, Y, G2, B2, G3, B3,
    N, scale, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=float("-inf")).to(tl.float32)

    # ---- softmax (fp32, like PyTorch's internal upcast) ----
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    p = e / s
    # round to bf16 (softmax output dtype in reference)
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- scale ----
    p = p * scale
    p = p.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 1 ----
    pm = tl.where(mask, p, 0.0)
    mean1 = tl.sum(pm, 0) / N
    d1 = tl.where(mask, p - mean1, 0.0)
    var1 = tl.sum(d1 * d1, 0) / N
    rstd1 = 1.0 / tl.sqrt(var1 + eps)
    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d1 * rstd1 * g2 + b2
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 ----
    ym = tl.where(mask, y, 0.0)
    mean2 = tl.sum(ym, 0) / N
    d2 = tl.where(mask, y - mean2, 0.0)
    var2 = tl.sum(d2 * d2, 0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = d2 * rstd2 * g3 + b3

    tl.store(Y + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = x * 1.4033
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_softmax_double_ln[(rows,)](
            x2, y,
            self.ln2_g, self.ln2_b, self.ln3_g, self.ln3_b,
            N, 1.4033, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
