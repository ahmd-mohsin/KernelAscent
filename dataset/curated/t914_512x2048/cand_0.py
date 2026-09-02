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
    X, RW, G, B, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (compute in fp32, cast to fp16, multiply by weight in fp16)
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (xf * rstd).to(tl.float16)
    rw = tl.load(RW + cols, mask=mask, other=0.0)
    y16 = y16 * rw  # fp16 multiply, matching reference

    # LayerNorm (fp32 accumulation, matching F.layer_norm on fp16)
    yf = y16.to(tl.float32)
    mean = tl.sum(yf, axis=0) / N
    d = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (yf - mean) * inv * g + b

    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


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
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_rms_ln_kernel[(m,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, out,
            x.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
