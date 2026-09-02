import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 982
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_rms_rms_bias_ln_kernel(
    X, OUT, W1, W2, B3, G4, B4,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # load fp16 row
    x = tl.load(X + row * D + offs, mask=mask, other=0.0)

    # relu (fp16)
    zero16 = tl.zeros_like(x)
    x = tl.maximum(x, zero16)

    # RMSNorm 1: compute in fp32, cast to fp16, multiply by w1 in fp16
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    x = (xf * r).to(tl.float16) * w1

    # RMSNorm 2
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    x = (xf * r).to(tl.float16) * w2

    # bias add (fp16)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0)
    x = x + b3

    # LayerNorm: fp32 math, biased variance, eps=1e-5
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / D
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv = tl.math.rsqrt(var + 1e-5)
    g4 = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b4 = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv * g4 + b4

    tl.store(OUT + row * D + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            return self._forward_ref(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_rms_rms_bias_ln_kernel[(m,)](
            x2d, out,
            self.rms1_w, self.rms2_w, self.b3, self.ln4_g, self.ln4_b,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

    def _forward_ref(self, x):
        x = torch.relu(x)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        x = x + self.b3
        x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
        return x
