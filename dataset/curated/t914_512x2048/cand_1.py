import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 914
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_rms_ln_kernel(
    X, W_RMS, G, B, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16, then fp16 multiply by weight)
    ms = tl.sum(xf * xf, axis=0) / N
    rstd_rms = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * rstd_rms).to(tl.float16)
    w = tl.load(W_RMS + cols, mask=mask, other=0.0)  # fp16
    h16 = y16 * w  # fp16 arithmetic to match reference
    h = h16.to(tl.float32)

    # LayerNorm in fp32
    mean = tl.sum(tl.where(mask, h, 0.0), axis=0) / N
    diff = tl.where(mask, h - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (h - mean) * rstd * g + b
    tl.store(Y + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        _fused_rms_ln_kernel[(m,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, y,
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return y
