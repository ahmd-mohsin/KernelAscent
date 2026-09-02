import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 169
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_gelu_rms_bias_kernel(
    X, W, B2, B3, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf variant), computed in fp32 then rounded to fp16
    # to match PyTorch's fp16 gelu output
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # RMSNorm in fp32 (matches reference: cast to float, mean of squares)
    gf = g16.to(tl.float32)
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    y16 = (gf * inv).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0)

    # fp16 arithmetic to match reference elementwise ops
    out = y16 * w
    out = out + b2
    out = out + b3

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xg = F.gelu(x)
            _xf = xg.float()
            xn = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xg.dtype) * self.rms1_w
            return xn + self.b2 + self.b3

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_gelu_rms_bias_kernel[(m,)](
            x2d, self.rms1_w, self.b2, self.b3, y,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
