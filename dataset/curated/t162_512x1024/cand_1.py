import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 162
M, D, DT = 512, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, Y, B1, W2, G3, B3, B4,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # bf16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)                 # bf16

    # relu + bias add in bf16 (matches reference dtype behavior)
    zero = tl.zeros_like(x)
    x = tl.maximum(x, zero)
    x = x + b1

    # RMSNorm in fp32
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = tl.math.rsqrt(ms + 1e-6)
    xn = (xf * rstd).to(tl.bfloat16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)  # bf16
    y = xn * w2                                    # bf16 multiply

    # LayerNorm in fp32 (as PyTorch does for bf16)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / N
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)

    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (diff * inv) * g3 + b3
    out_bf = out.to(tl.bfloat16)

    b4 = tl.load(B4 + cols, mask=mask, other=0.0)  # bf16
    out_bf = out_bf + b4

    tl.store(Y + row * stride_y + cols, out_bf, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x2, y, self.b1, self.rms2_w, self.ln3_g, self.ln3_b, self.b4,
            x2.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
