import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 98
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_post_kernel(
    X, W2, B4, W5, Out,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0)  # fp16

    # x = x * 1.1988 (PyTorch half kernels compute in fp32 opmath, round to fp16)
    x = (x.to(tl.float32) * 1.1988).to(tl.float16)

    # RMSNorm #1 (stats in fp32, then cast to fp16, multiply by weight)
    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    x = ((xf * r).to(tl.float16).to(tl.float32) * w2.to(tl.float32)).to(tl.float16)

    # ReLU
    x = tl.maximum(x, tl.zeros_like(x))

    # + b4
    b4 = tl.load(B4 + offs, mask=mask, other=0.0)
    x = (x.to(tl.float32) + b4.to(tl.float32)).to(tl.float16)

    # RMSNorm #2
    xf = x.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    w5 = tl.load(W5 + offs, mask=mask, other=0.0)
    y = ((xf * r2).to(tl.float16).to(tl.float32) * w5.to(tl.float32)).to(tl.float16)

    tl.store(Out + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        x = x @ self.W0
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_post_kernel[(m,)](
            x, self.rms2_w, self.b4, self.rms5_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
