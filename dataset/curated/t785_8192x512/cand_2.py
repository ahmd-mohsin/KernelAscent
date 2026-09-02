import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 785
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _softmax_rms_kernel(X, W, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax which accumulates in fp32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to bf16 (softmax output dtype), then back to fp32 for RMS norm
    p_bf = p.to(tl.bfloat16)
    xf = p_bf.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)

    normed = (xf * inv).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.bfloat16)
    out = normed * w

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_rms_kernel[(Mrows,)](
            h, self.rms2_w, y, N,
            h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
