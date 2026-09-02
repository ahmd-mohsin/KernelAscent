import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 476
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_scale_softmax2_ln_kernel(
    X, OUT, G, B,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # scale (rounded to bf16 as in reference)
    xs = (xf * SCALE).to(tl.bfloat16).to(tl.float32)

    # softmax 1 (fp32 math, bf16 rounding of output)
    m1 = tl.max(tl.where(mask, xs, float('-inf')), axis=0)
    e1 = tl.exp(xs - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # softmax 2
    m2 = tl.max(tl.where(mask, y1, float('-inf')), axis=0)
    e2 = tl.exp(y1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = (e2 / s2).to(tl.bfloat16).to(tl.float32)

    # layer norm (fp32 math)
    mean = tl.sum(y2, axis=0) / N
    diff = tl.where(mask, y2 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y2 - mean) * rstd * g + b

    tl.store(OUT + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # bf16 tensor-core matmul
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_scale_softmax2_ln_kernel[(Mrows,)](
            h, out, self.ln4_g, self.ln4_b,
            N, h.stride(0), out.stride(0),
            EPS=1e-5, SCALE=1.0685, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
