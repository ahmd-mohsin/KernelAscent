import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 620
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_epilogue(
    Y, OUT, B1, LN_G, LN_B, B4, RMS_W,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * N + cols, mask=mask, other=0.0)  # fp16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)

    # bias add in fp16 (match reference fp16 add)
    x = (y + b1).to(tl.float16)

    # gelu (exact, erf) computed in fp32 (PyTorch opmath), stored fp16
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = g.to(tl.float16)

    # layernorm in fp32, output fp16
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    ln_g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    ln_b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    x = ((xf - mean) * rstd * ln_g + ln_b).to(tl.float16)

    # bias add in fp16
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    x = (x + b4).to(tl.float16)

    # rmsnorm: fp32 stats, cast to fp16, multiply by weight in fp16
    xf = x.to(tl.float32)
    xf_m = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf_m * xf_m, axis=0) / N
    rrms = tl.math.rsqrt(ms + 1e-6)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0)
    out = ((xf * rrms).to(tl.float16) * w).to(tl.float16)

    tl.store(OUT + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        _fused_epilogue[(m,)](
            y, out, self.b1, self.ln3_g, self.ln3_b, self.b4, self.rms5_w,
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return out
