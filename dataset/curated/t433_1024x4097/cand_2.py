import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 433
M, D, DT = 1024, 4097, torch.bfloat16


@triton.jit
def _fused_rms_softmax_gelu_kernel(
    X, W, Y,
    N,
    stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, rounded to bf16, then * w in fp32 opmath, rounded to bf16)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32, output rounded to bf16
    t = tl.where(mask, y.to(tl.float32), float("-inf"))
    mmax = tl.max(t, axis=0)
    e = tl.exp(t - mmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16)

    # exact gelu in fp32 opmath, rounded to bf16
    pf = p.to(tl.float32)
    g = (pf * 0.5 * (1.0 + tl.math.erf(pf * 0.7071067811865476))).to(tl.bfloat16)

    # scalar mul in fp32 opmath, rounded to bf16
    out = (g.to(tl.float32) * 1.158).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_softmax_gelu_kernel[(m,)](
            x, self.rms0_w, y,
            n,
            x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y
