import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 372
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _fused_epilogue(Y, W, Out,
                    N,
                    stride_y, stride_o,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- load matmul output row (fp16) ----
    y = tl.load(Y + row * stride_y + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.where(mask, y, float('-inf'))

    # ---- softmax #1 (fp32 compute, fp16 output, matching torch half softmax) ----
    m1 = tl.max(y, axis=0)
    e1 = tl.exp(y - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = e1 / tl.sum(e1, axis=0)
    s1_h = s1.to(tl.float16)

    # relu is identity here (softmax output >= 0)

    # ---- softmax #2 ----
    x2 = s1_h.to(tl.float32)
    x2 = tl.where(mask, x2, float('-inf'))
    m2 = tl.max(x2, axis=0)
    e2 = tl.exp(x2 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = e2 / tl.sum(e2, axis=0)
    s2_h = s2.to(tl.float16)

    # ---- scale by 1.2112 (torch half scalar-mul uses fp32 opmath, then casts) ----
    z_h = (s2_h.to(tl.float32) * 1.2112).to(tl.float16)

    # ---- RMSNorm in fp32, cast to fp16, multiply by weight (fp16) ----
    xf = z_h.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / N
    r = xf * tl.math.rsqrt(ms + 1e-6)
    r_h = r.to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    out = r_h * w
    tl.store(Out + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            y = x @ self.W0
            y = torch.softmax(y, dim=-1)
            y = torch.relu(y)
            y = torch.softmax(y, dim=-1)
            y = y * 1.2112
            yf = y.float()
            y = (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms5_w
            return y

        # cuBLAS fp16 matmul (fp32 accumulate) -> tensor cores on A100
        y = torch.matmul(x, self.W0)
        if not y.is_contiguous():
            y = y.contiguous()

        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_epilogue[(m,)](
            y, self.rms5_w, out,
            n,
            y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
