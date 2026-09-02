import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 100
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(
    X, W, B2, G, B3, OUT,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # RMSNorm (fp32 compute, round to fp16 like reference)
    ms = tl.sum(xf * xf, axis=0) / D_
    rms = tl.math.rsqrt(ms + 1e-6)
    t = (xf * rms).to(tl.float16)

    # * w  (fp32 opmath, round to fp16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    t = (t.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # * 1.0401
    t = (t.to(tl.float32) * 1.0401).to(tl.float16)

    # + b2
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    t = (t.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)

    # LayerNorm in fp32
    tf = t.to(tl.float32)
    mean = tl.sum(tl.where(mask, tf, 0.0), axis=0) / D_
    diff = tl.where(mask, tf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D_
    inv = tl.math.rsqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (tf - mean) * inv * g + b3

    # ReLU
    y = tl.maximum(y, 0.0).to(tl.float16)
    tl.store(OUT + row * D_ + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x2, self.rms0_w, self.b2, self.ln3_g, self.ln3_b, out,
            d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
