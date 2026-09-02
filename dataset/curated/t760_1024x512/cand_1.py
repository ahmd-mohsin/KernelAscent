import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 760
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _softmax_gelu_ln_kernel(
    X, Y, G, B,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulate, round to fp16 like PyTorch output)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16).to(tl.float32)

    # exact gelu (erf), round to fp16 like PyTorch output
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * sm * (1.0 + tl.math.erf(sm * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # layernorm in fp32
    g = tl.where(mask, g, 0.0)
    mean = tl.sum(g, axis=0) / N
    diff = tl.where(mask, g - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (g - mean) * rstd * gamma + beta

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_gelu_ln_kernel[(Mrows,)](
            h, out, self.ln3_g, self.ln3_b,
            h.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK, EPS=1e-5,
            num_warps=8,
        )
        return out
