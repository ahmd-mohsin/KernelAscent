import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 37
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_gelu_relu_ln_scale(
    X, G, B, Y,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU in fp32 (matches PyTorch opmath), rounded to fp16 like F.gelu output
    inv_sqrt2 = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    g = g.to(tl.float16)

    # relu (fp16), then back to fp32 for layernorm stats (PyTorch accumulates in fp32)
    r = tl.maximum(g, tl.zeros_like(g)).to(tl.float32)
    r = tl.where(mask, r, 0.0)

    mean = tl.sum(r, axis=0) / N
    diff = tl.where(mask, r - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    gw = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bw = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    ln = (r - mean) * rstd * gw + bw
    ln_h = ln.to(tl.float16)  # layer_norm output rounded to fp16

    out = (ln_h.to(tl.float32) * scale).to(tl.float16)  # scalar mul in opmath fp32
    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_relu_ln_scale[(m,)](
            h, self.ln3_g, self.ln3_b, y,
            n, 1e-5, 1.4798,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
