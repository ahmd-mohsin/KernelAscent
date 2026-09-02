import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 139
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(X, W, Y, D, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- RMSNorm (fp32 math, matching reference) ----
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = (x * rstd).to(tl.float16)  # cast to fp16 like .to(x.dtype)

    w = tl.load(W + offs, mask=mask, other=0.0)  # fp16 weight
    v16 = xn * w                                  # fp16 multiply (matches half*half)
    v = v16.to(tl.float32)

    # ---- Softmax #1 (fp32 accum, fp16 output like torch half softmax) ----
    v = tl.where(mask, v, float('-inf'))
    m1 = tl.max(v, axis=0)
    e1 = tl.exp(v - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- Softmax #2 ----
    p = tl.where(mask, p, float('-inf'))
    m2 = tl.max(p, axis=0)
    e2 = tl.exp(p - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    q = (e2 / s2).to(tl.float16).to(tl.float32)

    # ---- GELU (erf-based, fp32 math like torch half gelu) ----
    out = 0.5 * q * (1.0 + tl.math.erf(q * 0.7071067811865476))

    tl.store(Y + row * D + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            x = torch.softmax(x, dim=-1)
            return F.gelu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.rms0_w, y, d, 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
