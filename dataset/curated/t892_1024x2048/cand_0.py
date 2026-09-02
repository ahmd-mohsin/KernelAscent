import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 892
M, D, DT = 1024, 2048, torch.float16

_SQRT1_2 = 0.7071067811865476


@triton.jit
def _fused_kernel(X, W1, W4, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    xptr = X + row * D + offs

    x = tl.load(xptr).to(tl.float32)

    # gelu (exact, opmath fp32 -> cast fp16 like PyTorch)
    g1 = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g1 = g1.to(tl.float16).to(tl.float32)

    # rmsnorm 1
    ms = tl.sum(g1 * g1, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w1 = tl.load(W1 + offs)
    y1 = (g1 * r).to(tl.float16) * w1  # fp16 multiply like PyTorch

    # gelu 2
    y1f = y1.to(tl.float32)
    g2 = 0.5 * y1f * (1.0 + tl.math.erf(y1f * 0.7071067811865476))
    g2 = g2.to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulate, fp16 output like PyTorch)
    mx = tl.max(g2, axis=0)
    e = tl.exp(g2 - mx)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # rmsnorm 2
    ms2 = tl.sum(p * p, axis=0) / D
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    w4 = tl.load(W4 + offs)
    out = (p * r2).to(tl.float16) * w4

    tl.store(Y + row * D + offs, out)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference implementation
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        _fused_kernel[(m,)](
            x, self.rms1_w, self.rms4_w, y,
            D=d, BLOCK=d,
            num_warps=8,
        )
        return y
