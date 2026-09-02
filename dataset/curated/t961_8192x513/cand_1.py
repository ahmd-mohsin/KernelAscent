import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 961
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_relu_rms_gelu(X, W, Y, D_: tl.constexpr, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    # ReLU (in input dtype, then upcast to fp32 like the reference)
    x = tl.where(x > 0, x, x * 0)
    xf = x.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(xf * xf, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    norm = xf * inv

    # cast normalized value to bf16 (matches .to(x.dtype)), then multiply by weight.
    # bf16*bf16 product is exact in fp32, so fp32 multiply + round == bf16 multiply.
    norm_b = norm.to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    prod = (norm_b * w).to(tl.bfloat16).to(tl.float32)

    # exact GELU in fp32 (PyTorch bf16 gelu computes in fp32 opmath)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * prod * (1.0 + tl.math.erf(prod * INV_SQRT2))

    tl.store(Y + row * stride + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xr = torch.relu(x)
            _xf = xr.float()
            xr = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.gelu(xr)

        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_relu_rms_gelu[(m,)](
            x, self.rms1_w, y, d, x.stride(0), BLOCK=BLOCK,
            num_warps=8,
        )
        return y
