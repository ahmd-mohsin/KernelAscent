import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 212
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_kernel(
    X, B0, G, B, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0)

    # x = x + b0 (fp16 add semantics: fp32 compute, round to fp16)
    x = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.float16)
    # x = x * 1.4934 (fp32 compute, round to fp16)
    x = (x.to(tl.float32) * 1.4934).to(tl.float16)

    # layer norm (compute in fp32)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((xf - mean) * rstd * g + b).to(tl.float16)

    # rms norm: fp32 on the fp16 layernorm output
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    z = (yf * rrms).to(tl.float16)

    # multiply by rms3_w (fp16 mul: fp32 compute, round to fp16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (z.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](
            x, self.b0, self.ln2_g, self.ln2_b, self.rms3_w, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
