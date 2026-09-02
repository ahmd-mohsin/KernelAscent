import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 239
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_epilogue(X, W3, W4, B5, Out, stride, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * stride + offs

    # load matmul output (fp16) and compute in fp32
    x = tl.load(ptr).to(tl.float32)

    # exact (erf) GELU, computed in fp32 as PyTorch does for half, rounded to fp16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax over the row, fp32 accumulation, rounded back to fp16
    mx = tl.max(g, axis=0)
    e = tl.exp(g - mx)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16)

    # RMSNorm 1: float math, cast to fp16, multiply by weight in fp16
    xf = p.to(tl.float32)
    r = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    w3 = tl.load(W3 + offs)
    x1 = (xf * r).to(tl.float16) * w3

    # RMSNorm 2
    xf = x1.to(tl.float32)
    r = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / N + 1e-6)
    w4 = tl.load(W4 + offs)
    x2 = (xf * r).to(tl.float16) * w4

    # bias add in fp16
    b = tl.load(B5 + offs)
    out = x2 + b

    tl.store(Out + row * stride + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        rows, n = y.shape
        out = torch.empty_like(y)
        _fused_epilogue[(rows,)](
            y, self.rms3_w, self.rms4_w, self.b5, out,
            y.stride(0), N=n, BLOCK=n,
            num_warps=4,
        )
        return out
