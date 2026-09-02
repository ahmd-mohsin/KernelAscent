import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 853
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_gelu_ln_softmax(
    X, G, B, Y,
    N,  # row length
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # scale (emulate bf16 rounding of the intermediate, matching eager execution)
    x = x * 1.3715
    x = x.to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # layer norm (fp32 accumulation, biased variance)
    g_masked = tl.where(mask, g, 0.0)
    mean = tl.sum(g_masked, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * gamma + beta
    y = y.to(tl.bfloat16).to(tl.float32)

    # softmax
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * N + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.3715
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_ln_softmax[(rows,)](
            x2, self.ln2_g, self.ln2_b, out,
            n,
            EPS=1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
