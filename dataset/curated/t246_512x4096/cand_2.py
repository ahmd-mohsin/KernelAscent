import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 246
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _fused_post_kernel(X, W, B, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- softmax #1 (fp32 internals, result rounded to fp16 like torch) ----
    x = tl.load(X + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m1 = tl.max(x, 0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, 0)
    sm = (e1 / s1).to(tl.float16)

    # ---- RMSNorm in fp32 on the fp16-rounded softmax output ----
    xf = sm.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), 0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * r).to(tl.float16)

    # ---- scale (fp16), bias (fp16), relu ----
    w = tl.load(W + offs, mask=mask, other=0.0)
    b = tl.load(B + offs, mask=mask, other=0.0)
    y = xn * w          # fp16 multiply, matching torch fp16 * fp16
    y = y + b           # fp16 add
    y = tl.maximum(y, y * 0)  # relu in fp16

    # ---- softmax #2 (fp32 internals) ----
    yf = tl.where(mask, y.to(tl.float32), float('-inf'))
    m2 = tl.max(yf, 0)
    e2 = tl.exp(yf - m2)
    s2 = tl.sum(tl.where(mask, e2, 0.0), 0)
    out = (e2 / s2).to(tl.float16)

    tl.store(Out + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(rows,)](
            h, self.rms2_w, self.b3, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
