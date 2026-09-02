import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 45
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_bias_rms_kernel(X, B1, B2, W, B4, Out,
                           N, stride_x, stride_o,
                           EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)   # fp16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)                  # fp16
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)                  # fp16

    x = x + b1          # fp16 add (matches torch fp16 add)
    x = x + b2          # fp16 add

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + EPS)

    xn = (xf * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)    # fp16
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)  # fp16

    y = xn * w + b4  # fp16 mul then fp16 add
    tl.store(Out + row * stride_o + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (already optimal on A100 tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_bias_rms_kernel[(Mrows,)](
            h, self.b1, self.b2, self.rms3_w, self.b4, out,
            N, h.stride(0), out.stride(0),
            EPS=1e-6, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
