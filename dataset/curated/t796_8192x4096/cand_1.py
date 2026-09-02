import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 796
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _softmax_rmsnorm_kernel(X, W, Y, stride_x, stride_y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to bf16 (softmax output dtype), then back to fp32 (x.float())
    pf = p.to(tl.bfloat16).to(tl.float32)

    # rmsnorm
    ms = tl.sum(tl.where(mask, pf * pf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (pf * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)
    tl.store(Y + row * stride_y + cols, out, mask=mask)


@triton.jit
def _softmax_layernorm_kernel(X, G, B, Y, stride_x, stride_y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to bf16 (softmax output dtype), then back to fp32 for layernorm
    pf = p.to(tl.bfloat16).to(tl.float32)

    # layernorm (eps = 1e-5)
    mean = tl.sum(tl.where(mask, pf, 0.0), axis=0) / N
    d = tl.where(mask, pf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = ((pf - mean) * rstd * g + b).to(tl.bfloat16)
    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        BLOCK = triton.next_power_of_2(N)

        y1 = torch.empty_like(x)
        _softmax_rmsnorm_kernel[(Mrows,)](
            x, self.rms1_w, y1,
            x.stride(0), y1.stride(0),
            N, BLOCK=BLOCK, num_warps=16,
        )

        z = y1 @ self.W2

        out = torch.empty_like(z)
        _softmax_layernorm_kernel[(Mrows,)](
            z, self.ln4_g, self.ln4_b, out,
            z.stride(0), out.stride(0),
            N, BLOCK=BLOCK, num_warps=16,
        )
        return out
