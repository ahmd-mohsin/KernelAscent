import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 280
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_rms_bias_ln_kernel(
    X_ptr, RW_ptr, B2_ptr, G_ptr, B3_ptr, Y_ptr,
    N,
    EPS_RMS: tl.constexpr,
    EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm (computed in fp32, cast to fp16 before weight multiply, like reference)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + EPS_RMS)
    xh = (xf * rrms).to(tl.float16)

    rw = tl.load(RW_ptr + offs, mask=mask, other=0.0)
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0)

    # fp16 multiply then fp16 add (match reference dtype behavior)
    t = (xh * rw).to(tl.float16)
    y = (t + b2).to(tl.float16)

    # LayerNorm in fp32
    yf = y.to(tl.float32)
    mean = tl.sum(yf, axis=0) / N
    d = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)

    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    out = (d * rstd) * g + b3
    tl.store(Y_ptr + row * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_bias_ln_kernel[(Mrows,)](
            x, self.rms1_w, self.b2, self.ln3_g, self.ln3_b, y,
            N,
            EPS_RMS=1e-6, EPS_LN=1e-5,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y @ self.W4
