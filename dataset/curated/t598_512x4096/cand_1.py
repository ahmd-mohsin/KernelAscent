import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 598
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_kernel(X, G, B, W, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    # load + relu (fp32 math)
    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # layer norm (fp32, biased variance, eps=1e-5) — matches F.layer_norm on fp16
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    # cast to fp16 (layer_norm output dtype), then relu
    y = y.to(tl.float16).to(tl.float32)
    y = tl.maximum(y, 0.0)

    # softmax in fp32 (as PyTorch does for fp16 inputs), output rounded to fp16
    y = tl.where(mask, y, float('-inf'))
    mx = tl.max(y, axis=0)
    e = tl.exp(y - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # RMS norm in fp32 (eps=1e-6), cast to fp16, then multiply by weight
    ms = tl.sum(p * p, axis=0) / N
    r = p * tl.math.rsqrt(ms + 1e-6)
    r = r.to(tl.float16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (r * w).to(tl.float16)

    tl.store(Y + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(rows,)](
            x2, self.ln1_g, self.ln1_b, self.rms4_w, y, N,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view(orig_shape)
