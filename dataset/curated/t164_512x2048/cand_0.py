import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 164
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_norms_kernel(
    X_ptr, W1_ptr, W2_ptr, G_ptr, B_ptr, Y_ptr,
    N,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm 1 (round to bf16 before weight mult, like reference)
    ms1 = tl.sum(xf * xf, axis=0) / N
    y = (xf * tl.math.rsqrt(ms1 + EPS_RMS)).to(tl.bfloat16)
    w1 = tl.load(W1_ptr + offs, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w1.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm 2
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    z = (yf * tl.math.rsqrt(ms2 + EPS_RMS)).to(tl.bfloat16)
    w2 = tl.load(W2_ptr + offs, mask=mask, other=0.0)
    z = (z.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # scale by 1.0772 (fp32 compute, round to bf16)
    z = (z.to(tl.float32) * SCALE).to(tl.bfloat16)

    # LayerNorm (fp32 internals, bf16 output)
    zf = z.to(tl.float32)
    zf_m = tl.where(mask, zf, 0.0)
    mean = tl.sum(zf_m, axis=0) / N
    diff = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + EPS_LN)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = ((zf - mean) * rstd * g + b).to(tl.bfloat16)

    tl.store(Y_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norms_kernel[(rows,)](
            x, self.rms1_w, self.rms2_w, self.ln4_g, self.ln4_b, y,
            N,
            EPS_RMS=1e-6,
            EPS_LN=1e-5,
            SCALE=1.0772,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y @ self.W5
