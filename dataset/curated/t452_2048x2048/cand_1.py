import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 452
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, W2, W3, G, B, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # scale by 1.499 (rounded to bf16 as in reference)
    x = (x * 1.499).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 3
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(W3 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w3).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (fp32 internal, eps=1e-5)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * inv * g + b

    tl.store(Y + row * N + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # tensor-core matmul
        x = torch.matmul(x, self.W0)
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_norm_kernel[(m,)](
            x, self.rms2_w, self.rms3_w, self.ln4_g, self.ln4_b, y,
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
