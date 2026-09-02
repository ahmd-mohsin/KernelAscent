import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 281
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, W0, W1, B3, OUT,
    n_cols,
    stride_x, stride_out,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm 0
    ms = tl.sum(xf * xf, axis=0) / n_cols
    r = 1.0 / tl.sqrt(ms + eps)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf * r).to(tl.bfloat16).to(tl.float32)
    y = (y * w0).to(tl.bfloat16)

    # RMSNorm 1
    yf = y.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / n_cols
    r2 = 1.0 / tl.sqrt(ms2 + eps)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (yf * r2).to(tl.bfloat16).to(tl.float32)
    z = (z * w1).to(tl.bfloat16)

    # ReLU
    z = tl.maximum(z, 0.0)

    # Add bias (computed in fp32, rounded to bf16, like PyTorch)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z.to(tl.float32) + b3).to(tl.bfloat16)

    # Softmax (fp32 accumulation)
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    zmax = tl.max(zf, axis=0)
    num = tl.exp(zf - zmax)
    num = tl.where(mask, num, 0.0)
    den = tl.sum(num, axis=0)
    out = (num / den).to(tl.bfloat16)

    tl.store(OUT + row * stride_out + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(n_rows,)](
            x, self.rms0_w, self.rms1_w, self.b3, out,
            n_cols,
            x.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
