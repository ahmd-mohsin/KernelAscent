import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 561
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _rms_gelu_kernel(
    X, W, Out,
    N, stride_x, stride_o,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (mean of squares in fp32)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = x * rstd

    # cast to bf16 then multiply by weight (match reference: .to(dtype) * w)
    xn = xn.to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w

    # exact GELU (erf-based), computed as F.gelu would on bf16 input (internally fp32 math)
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # final scale: bf16 * scalar
    out = (g.to(tl.float32) * scale).to(tl.bfloat16)

    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _rms_gelu_kernel[(m,)](
            x, self.rms1_w, out,
            n, x.stride(0), out.stride(0),
            1e-6, 1.3603,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
