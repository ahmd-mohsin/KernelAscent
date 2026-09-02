import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 144
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_rms_relu_softmax(X, W, Out, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0)  # fp16

    # x = x * 1.2328  (PyTorch half elementwise: compute in fp32, round to fp16)
    xh = (x.to(tl.float32) * 1.2328).to(tl.float16)

    # RMSNorm in fp32
    xf = xh.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    y = (xf * inv).to(tl.float16)

    # * rms2_w  (half*half -> fp32 opmath -> fp16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # * 1.4636
    y = (y.to(tl.float32) * 1.4636).to(tl.float16)

    # relu
    y = tl.maximum(y, 0.0)

    # softmax (fp32 accumulate, as PyTorch does for half)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core GEMM
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_rms_relu_softmax[(m,)](
            h, self.rms2_w, out, n,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return out
