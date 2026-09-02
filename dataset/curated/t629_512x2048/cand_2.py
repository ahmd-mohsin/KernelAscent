import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 629
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_softmax_rms_bias(
    X, W, B3, B4, Out,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches torch's fp16 softmax which computes in fp32)
    row_max = tl.max(x, axis=0)
    e = tl.math.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    # cast to fp16 (reference casts back to fp16 after softmax)
    sm16 = sm.to(tl.float16)

    # RMSNorm in fp32 on fp16 values
    xf = sm16.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    y16 = (xf * r).to(tl.float16)

    # fp16 elementwise: * w + b3 + b4
    w = tl.load(W + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)

    out = (y16 * w + b3) + b4
    tl.store(Out + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_softmax_rms_bias[(m,)](
            h, self.rms2_w, self.b3, self.b4, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=16,
        )
        return out
