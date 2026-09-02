import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 361
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_ln_softmax_ln_act(
    X, Y, G1, B1, G3, B3,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm 1 (fp32 math, round to bf16 like PyTorch op boundary)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g1 = tl.load(G1 + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    h = xc * rstd * g1 + b1
    h = h.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulation)
    h_masked = tl.where(mask, h, float('-inf'))
    m = tl.max(h_masked, axis=0)
    e = tl.exp(h_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.bfloat16).to(tl.float32)

    # LayerNorm 2
    mean2 = tl.sum(p, axis=0) / N
    pc = tl.where(mask, p - mean2, 0.0)
    var2 = tl.sum(pc * pc, axis=0) / N
    rstd2 = 1.0 / tl.sqrt(var2 + eps)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z = pc * rstd2 * g3 + b3
    z = z.to(tl.bfloat16).to(tl.float32)

    # ReLU
    z = tl.maximum(z, 0.0)
    z = z.to(tl.bfloat16).to(tl.float32)

    # GELU (exact, erf-based)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * z * (1.0 + tl.math.erf(z * INV_SQRT2))

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        M_, N_ = h.shape
        y = torch.empty_like(h)
        _fused_ln_softmax_ln_act[(M_,)](
            h, y,
            self.ln1_g, self.ln1_b, self.ln3_g, self.ln3_b,
            N_, h.stride(0), y.stride(0),
            1e-5,
            BLOCK=triton.next_power_of_2(N_),
            num_warps=8,
        )
        return y
