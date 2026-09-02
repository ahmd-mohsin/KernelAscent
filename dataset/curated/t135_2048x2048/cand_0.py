import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 135
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_rows_kernel(
    X_ptr, W2_ptr, W4_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)

    # ReLU (fp16)
    x = tl.maximum(x, 0.0)

    # RMSNorm #1 (fp32 accumulation)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + eps)
    w2 = tl.load(W2_ptr + cols, mask=mask, other=0.0)
    y = (xf * r).to(tl.float16) * w2  # fp16 multiply, matches PyTorch

    # Softmax (fp32 accumulation like PyTorch half softmax)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float("-inf"))
    mmax = tl.max(yf, axis=0)
    e = tl.exp(yf - mmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    o = (e / s).to(tl.float16)

    # RMSNorm #2
    of = o.to(tl.float32)
    ms2 = tl.sum(of * of, axis=0) / N
    r2 = tl.math.rsqrt(ms2 + eps)
    w4 = tl.load(W4_ptr + cols, mask=mask, other=0.0)
    z = (of * r2).to(tl.float16) * w4

    # ReLU
    z = tl.maximum(z, 0.0)

    tl.store(Out_ptr + row * stride_o + cols, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_rows_kernel[(rows,)](
            h, self.rms2_w, self.rms4_w, out,
            N, h.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
