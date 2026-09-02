import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 259
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_bias_rms_gelu(X, B, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # x = x + b0  (PyTorch bf16 add: fp32 opmath, rounded to bf16)
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    xb = (x + b).to(tl.bfloat16)

    # RMS norm in fp32
    xf = xb.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * r).to(tl.bfloat16).to(tl.float32)

    # * rms1_w (fp32 opmath, rounded to bf16)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16).to(tl.float32)

    # exact GELU (fp32 opmath), round back to bf16
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    tl.store(Y + row * D + offs, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.gelu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        rows = xc.shape[0]
        y = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        _fused_bias_rms_gelu[(rows,)](
            xc, self.b0, self.rms1_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=4,
        )
        return y.view(orig_shape)
