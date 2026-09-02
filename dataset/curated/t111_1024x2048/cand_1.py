import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 111
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _gelu_softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # exact GELU (erf variant), computed in fp32 then rounded to bf16 (match reference)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


@triton.jit
def _ln_scale_kernel(X, G, B, Y, N, stride_x, stride_y, eps, scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # round to bf16 as layer_norm output, then scale (matches reference op order)
    y = y.to(tl.bfloat16).to(tl.float32) * scale
    tl.store(Y + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W2 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        sm = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _gelu_softmax_kernel[(Mrows,)](
            x, sm, N, x.stride(0), sm.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        h = sm @ self.W2
        N2 = h.shape[-1]
        out = torch.empty_like(h)
        BLOCK2 = triton.next_power_of_2(N2)
        _ln_scale_kernel[(Mrows,)](
            h, self.ln3_g, self.ln3_b, out, N2,
            h.stride(0), out.stride(0), 1e-5, 1.4031,
            BLOCK=BLOCK2, num_warps=16,
        )
        return out
