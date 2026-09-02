import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 87
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_ln_ln_relu_rms(
    X, Y, G0, B0, G1, B1, W,
    N, stride_x, stride_y,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, fp16 rounding like PyTorch) ----
    mean = tl.sum(x, axis=0) / D
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D
    inv = tl.rsqrt(var + 1e-5)
    g0 = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (d * inv * g0 + b0).to(tl.float16).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    # ---- LayerNorm 1 ----
    mean = tl.sum(x, axis=0) / D
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D
    inv = tl.rsqrt(var + 1e-5)
    g1 = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (d * inv * g1 + b1).to(tl.float16).to(tl.float32)

    # ---- ReLU ----
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, 0.0)

    # ---- RMSNorm (fp32 math, cast to fp16, then fp16 multiply by weight) ----
    ms = tl.sum(x * x, axis=0) / D
    r = tl.rsqrt(ms + 1e-6)
    xh = (x * r).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = xh * w

    tl.store(Y + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_ln_ln_relu_rms[(n,)](
            x2, y,
            self.ln0_g, self.ln0_b,
            self.ln1_g, self.ln1_b,
            self.rms3_w,
            n, x2.stride(0), y.stride(0),
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
