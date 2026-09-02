import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 350
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, G, B, W, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # x = x * 1.4563 (bf16 elementwise: compute in fp32, round to bf16)
    xs = (x.to(tl.float32) * 1.4563).to(tl.bfloat16)
    xf = xs.to(tl.float32)

    # LayerNorm (fp32 internals, bf16 output)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd) * g + b
    y_bf = y.to(tl.bfloat16)

    # GELU (exact, fp32 internals, bf16 output)
    yf = y_bf.to(tl.float32)
    gel = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    gel_bf = gel.to(tl.bfloat16)

    # RMSNorm in fp32, cast to bf16, then multiply by weight (fp32 opmath, bf16 out)
    gf = gel_bf.to(tl.float32)
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    r = gf * (1.0 / tl.sqrt(ms + 1e-6))
    r_bf = r.to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (r_bf.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x, self.ln1_g, self.ln1_b, self.rms3_w, y,
            n, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return y
