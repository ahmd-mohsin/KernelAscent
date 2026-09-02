import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 796
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(X, W, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to bf16 (matching torch.softmax output dtype), then back to fp32
    p = p.to(tl.bfloat16).to(tl.float32)
    # RMSNorm in fp32
    ms = tl.sum(p * p, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y = (p * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)
    tl.store(Y + row * stride_y + cols, out, mask=mask)


@triton.jit
def _softmax_ln_kernel(X, G, B, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    # softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    # round to bf16 (matching torch.softmax output dtype), then back to fp32
    p = p.to(tl.bfloat16).to(tl.float32)
    # layer_norm in fp32 (eps = 1e-5, biased variance)
    mean = tl.sum(p, axis=0) / N
    d = tl.where(mask, p - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (d * inv * g + b).to(tl.bfloat16)
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
        _softmax_rms_kernel[(Mrows,)](
            x, self.rms1_w, y1, N, x.stride(0), y1.stride(0),
            BLOCK=BLOCK, num_warps=16,
        )

        y2 = y1 @ self.W2

        out = torch.empty_like(y2)
        _softmax_ln_kernel[(Mrows,)](
            y2, self.ln4_g, self.ln4_b, out, N, y2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=16,
        )
        return out
