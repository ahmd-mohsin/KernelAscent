import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 86
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(X, B, W, Out, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * D + offs

    x = tl.load(X + base).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    # bias add (rounded to fp16 like the reference's fp16 add)
    x = (x + b).to(tl.float16).to(tl.float32)

    # softmax #1 (fp32 accumulation, fp16 output like torch half softmax)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32, cast to fp16, then fp16 multiply by weight
    ms = tl.sum(x * x, 0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (x * r).to(tl.float16)
    w = tl.load(W + offs)
    x = (xh * w).to(tl.float32)

    # softmax #2
    m2 = tl.max(x, 0)
    e2 = tl.exp(x - m2)
    s2 = tl.sum(e2, 0)
    y = (e2 / s2).to(tl.float16)

    tl.store(Out + base, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        rows, d = x.shape
        out = torch.empty_like(x)
        _fused_kernel[(rows,)](
            x, self.b0, self.rms2_w, out,
            D=d, BLOCK=d,
            num_warps=8,
        )
        return out
