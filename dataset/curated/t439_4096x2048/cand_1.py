import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 439
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(X, G, B, Y, N, eps, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf), then round to bf16 like the reference
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 stats, biased variance, eps=1e-5)
    mean = tl.sum(tl.where(mask, g, 0.0), axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g - mean) * rstd * w + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # ReLU
    y = tl.maximum(y, 0.0)

    # Softmax (fp32, max-subtracted)
    y_m = tl.where(mask, y, float('-inf'))
    mx = tl.max(y_m, axis=0)
    e = tl.exp(y_m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    sm = sm.to(tl.bfloat16).to(tl.float32)

    # scale, then final round to bf16
    out = (sm * scale).to(tl.bfloat16)
    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        orig_shape = x.shape
        N = x.shape[-1]
        x2 = x.view(-1, N)
        Mrows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(Mrows,)](
            x2, self.ln1_g, self.ln1_b, y,
            N, 1e-5, 1.4972,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
