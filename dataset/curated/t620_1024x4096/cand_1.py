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
    Y, OUT, B1, G3, B3, B4, W5,
    N, stride_y, stride_o,
    LN_EPS: tl.constexpr, RMS_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * stride_y + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b1 (fp16 rounding)
    x = (y + b1).to(tl.float16).to(tl.float32)

    # gelu (exact, erf), computed in fp32, rounded to fp16
    inv_sqrt2 = 0.7071067811865476
    x = (x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))).to(tl.float16).to(tl.float32)

    # layer_norm over N elements (fp32 accum)
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + LN_EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    x = ((x - mean) * inv_std * g3 + b3).to(tl.float16).to(tl.float32)

    # x = x + b4 (fp16 rounding)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b4).to(tl.float16).to(tl.float32)

    # rmsnorm in fp32, round to fp16, then multiply by weight (fp32 opmath, round fp16)
    xs = tl.where(mask, x * x, 0.0)
    ms = tl.sum(xs, axis=0) / N
    rrms = tl.math.rsqrt(ms + RMS_EPS)
    x = (x * rrms).to(tl.float16).to(tl.float32)
    w5 = tl.load(W5 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (x * w5).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


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
        y = x @ self.W0
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue[(m,)](
            y, out, self.b1, self.ln3_g, self.ln3_b, self.b4, self.rms5_w,
            n, y.stride(0), out.stride(0),
            LN_EPS=1e-5, RMS_EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
