import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 507
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_ln_gelu_softmax(
    X, OUT, G, B, BIAS,
    N, stride_x, stride_o,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch internally)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = diff * rstd * g + b
    # round to fp16 as PyTorch would between ops
    y = y.to(tl.float16)

    # scale and bias (fp16 ops in reference)
    y = (y * SCALE).to(tl.float16)
    bias = tl.load(BIAS + cols, mask=mask, other=0.0)
    y = (y + bias).to(tl.float16)

    # GELU (exact erf variant, computed in fp32 internally on half)
    yf = y.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gelu = 0.5 * yf * (1.0 + tl.math.erf(yf * INV_SQRT2))
    gelu = gelu.to(tl.float16).to(tl.float32)

    # Softmax (fp32 accumulation)
    gelu = tl.where(mask, gelu, float("-inf"))
    mx = tl.max(gelu, axis=0)
    e = tl.exp(gelu - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS tensor-core GEMM
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_ln_gelu_softmax[(m,)](
            x, out, self.ln1_g, self.ln1_b, self.b3,
            n, x.stride(0), out.stride(0),
            EPS=1e-5, SCALE=1.2202, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
