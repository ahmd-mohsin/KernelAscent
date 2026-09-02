import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 374
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_kernel(X, W, G, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, cast to fp16 like PyTorch)
    xmax = tl.max(x, axis=0)
    e = tl.exp(x - xmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # RMSNorm: fp32 math, cast to fp16, multiply by weight in fp16
    smf = sm.to(tl.float32)
    ms = tl.sum(smf * smf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    rms_out = (smf * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    h = (rms_out * w).to(tl.float16)  # fp16 multiply

    # LayerNorm: fp32 internal math, output fp16
    hf = h.to(tl.float32)
    mean = tl.sum(hf, axis=0) / N
    d = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b

    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, y,
            n, BLOCK=BLOCK, num_warps=8,
        )
        return y
