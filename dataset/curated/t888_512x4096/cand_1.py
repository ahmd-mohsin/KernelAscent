import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 888
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(X, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + offs).to(tl.float32)

    # x = x * 1.352 (computed in fp32, rounded to bf16 like PyTorch)
    x = (x * 1.352).to(tl.bfloat16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    # gelu #1 (exact erf gelu, fp32 math, round to bf16)
    x = (0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    # gelu #2
    x = (0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    # gelu #3
    x = (0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + offs).to(tl.float32)
    y = (xn * w).to(tl.bfloat16)
    tl.store(Y + row * D + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.352
            x = F.gelu(x)
            x = F.gelu(x)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        w = self.rms4_w
        _fused_kernel[(m,)](x, w, y, D=d, BLOCK=d, num_warps=8)
        return y
