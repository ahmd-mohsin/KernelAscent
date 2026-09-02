import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 758
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, W0, B1, W3, W4, Y, N: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    base = row * N

    x = tl.load(X + base + offs).to(tl.float32)

    # RMSNorm 0 (fp32 compute, round to bf16, then bf16-style mul by weight)
    rs = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + 1e-6)
    x = (x * rs).to(tl.bfloat16).to(tl.float32)
    w0 = tl.load(W0 + offs).to(tl.float32)
    x = (x * w0).to(tl.bfloat16).to(tl.float32)

    # bias add (bf16 rounding)
    b1 = tl.load(B1 + offs).to(tl.float32)
    x = (x + b1).to(tl.bfloat16).to(tl.float32)

    # exact GELU (compute fp32, round to bf16)
    x = (0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 3
    rs = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + 1e-6)
    x = (x * rs).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(W3 + offs).to(tl.float32)
    x = (x * w3).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 4
    rs = tl.math.rsqrt(tl.sum(x * x, axis=0) / N + 1e-6)
    x = (x * rs).to(tl.bfloat16).to(tl.float32)
    w4 = tl.load(W4 + offs).to(tl.float32)
    y = (x * w4).to(tl.bfloat16)

    tl.store(Y + base + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)
        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2d.shape
        y = torch.empty_like(x2d)
        _fused_kernel[(m,)](
            x2d, self.rms0_w, self.b1, self.rms3_w, self.rms4_w, y,
            N=n, num_warps=4,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = x + self.b1
        x = F.gelu(x)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
        return x
