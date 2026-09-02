import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 740
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_rms_softmax_ln_kernel(
    X, OUT, RW, G, B,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (compute in fp32, round to bf16 as in reference)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16)

    rw = tl.load(RW + cols, mask=mask, other=0.0)
    # bf16 elementwise mul in torch: fp32 opmath then round to bf16
    y = (xn.to(tl.float32) * rw.to(tl.float32)).to(tl.bfloat16)

    # softmax: fp32 computation from bf16 input, output bf16
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16)

    # layer_norm: fp32 computation from bf16 input, output bf16
    smf = sm.to(tl.float32)
    mean = tl.sum(tl.where(mask, smf, 0.0), axis=0) / N
    d = tl.where(mask, smf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = ((smf - mean) * rstd * g + b).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        m, n = x.shape
        out = torch.empty_like(x)
        _fused_rms_softmax_ln_kernel[(m,)](
            x, out, self.rms1_w, self.ln3_g, self.ln3_b,
            x.stride(0), out.stride(0),
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return out
