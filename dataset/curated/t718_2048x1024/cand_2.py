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
    X, Y,
    g0, b0, g2, b2,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 0 (fp32 math, like PyTorch for bf16 inputs)
    mean0 = tl.sum(x, axis=0) / N
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + EPS)
    w0 = tl.load(g0 + cols, mask=mask, other=0.0).to(tl.float32)
    c0 = tl.load(b0 + cols, mask=mask, other=0.0).to(tl.float32)
    h = d0 * rstd0 * w0 + c0
    # round to bf16 (intermediate dtype in reference)
    h = h.to(tl.bfloat16).to(tl.float32)

    # GELU (exact, erf-based) computed in fp32, rounded back to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    h = 0.5 * h * (1.0 + tl.math.erf(h * INV_SQRT2))
    h = h.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    hm = tl.where(mask, h, 0.0)
    mean2 = tl.sum(hm, axis=0) / N
    d2 = tl.where(mask, h - mean2, 0.0)
    var2 = tl.sum(d2 * d2, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + EPS)
    w2 = tl.load(g2 + cols, mask=mask, other=0.0).to(tl.float32)
    c2 = tl.load(b2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d2 * rstd2 * w2 + c2

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


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
        xc = x.contiguous().view(-1, N)
        rows = xc.shape[0]
        y = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_ln_gelu_ln[(rows,)](
            xc, y,
            self.ln0_g, self.ln0_b, self.ln2_g, self.ln2_b,
            N=N, BLOCK=BLOCK, EPS=1e-5,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
