import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 961
M, D, DT = 8192, 513, torch.bfloat16


@triton.jit
def _fused_relu_rms_gelu(X, W, Y, D, stride_x, stride_y, EPS: tl.constexpr,
                         BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    # ReLU (exact in bf16)
    x = tl.maximum(x, 0.0)

    # RMSNorm computed in fp32 (matches reference)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = tl.math.rsqrt(ms + EPS)
    xn = (xf * inv).to(tl.bfloat16)  # cast back to input dtype (bf16)

    # multiply by weight: bf16 * bf16 with fp32 opmath, round to bf16
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # exact GELU in fp32 from bf16 input, round to bf16
    yf = y.to(tl.float32)
    g = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y + row * stride_y + offs, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_relu_rms_gelu[(m,)](
            x, self.rms1_w, out,
            d, x.stride(0), out.stride(0),
            EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
