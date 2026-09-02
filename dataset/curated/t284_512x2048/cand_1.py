import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 284
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_gelu_bias_rms_kernel(
    X_ptr, B_ptr, W_ptr, Y_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32, rounded to bf16 (matches F.gelu on bf16)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale by 1.164, round to bf16
    g = (g * SCALE).to(tl.bfloat16).to(tl.float32)

    # add bias, round to bf16
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = (g + b).to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(tl.where(mask, v * v, 0.0), axis=0) / D
    r = tl.math.rsqrt(ms + EPS)
    n = (v * r).to(tl.bfloat16).to(tl.float32)

    # multiply by rms weight, output bf16
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (n * w).to(tl.bfloat16)
    tl.store(Y_ptr + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = y * 1.164
            y = y + self.b2
            _yf = y.float()
            y = (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms3_w
            return y

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_bias_rms_kernel[(rows,)](
            x2, self.b2, self.rms3_w, out,
            D=d, EPS=1e-6, SCALE=1.164,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
