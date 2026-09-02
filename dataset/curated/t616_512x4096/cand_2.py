import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 616
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_gelu_softmax_relu_ln(
    X, G, B, Y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based) computed in fp32 then rounded to fp16
    # (matches PyTorch half-precision opmath behavior)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax in fp32, output rounded to fp16 (matches PyTorch)
    g_masked = tl.where(mask, g, float("-inf"))
    m = tl.max(g_masked, 0)
    e = tl.where(mask, tl.exp(g_masked - m), 0.0)
    s = tl.sum(e, 0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # ReLU (identity on softmax output, kept for exactness)
    p = tl.maximum(p, 0.0)

    # LayerNorm: stats in fp32, affine in fp32, output fp16
    mean = tl.sum(tl.where(mask, p, 0.0), 0) / D
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, 0) / D
    rstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    y = d * rstd * gamma + beta
    tl.store(Y + row * D + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            x = torch.relu(x)
            return F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_gelu_softmax_relu_ln[(rows,)](
            x2, self.ln3_g, self.ln3_b, y,
            D=d, EPS=1e-5, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
