import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 629
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _softmax_rms_bias_kernel(
    X, W, B3, B4, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for fp16 inputs)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # round to fp16 (softmax output dtype), then RMSNorm in fp32
    p16 = p.to(tl.float16)
    pf = p16.to(tl.float32)
    ms = tl.sum(tl.where(mask, pf * pf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    normed = (pf * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)

    y = normed * w
    y = y + b3
    y = y + b4

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core GEMM
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _softmax_rms_bias_kernel[(Mrows,)](
            h, self.rms2_w, self.b3, self.b4, y,
            N, h.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
