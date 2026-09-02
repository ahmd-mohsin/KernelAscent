import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 139
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_rms_smax2_gelu(X, W, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- RMSNorm (fp32 math, cast to fp16, scale by fp16 weight) ----
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * r).to(tl.float16)
    w = tl.load(W + offs, mask=mask, other=0.0)
    v = (xn * w).to(tl.float32)

    # ---- softmax #1 (fp32 accumulation, fp16 round-trip like PyTorch) ----
    v = tl.where(mask, v, float('-inf'))
    m1 = tl.max(v, axis=0)
    e1 = tl.exp(v - m1)
    s1 = tl.sum(e1, axis=0)
    p = (e1 / s1).to(tl.float16).to(tl.float32)

    # ---- softmax #2 ----
    p = tl.where(mask, p, float('-inf'))
    m2 = tl.max(p, axis=0)
    e2 = tl.exp(p - m2)
    s2 = tl.sum(e2, axis=0)
    q = (e2 / s2).to(tl.float16).to(tl.float32)

    # ---- exact (erf-based) GELU in fp32 ----
    g = q * 0.5 * (1.0 + tl.math.erf(q * 0.7071067811865476))
    tl.store(Y + row * stride_y + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            # fallback: reference path
            _xf = x.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            y = torch.softmax(y, dim=-1)
            y = torch.softmax(y, dim=-1)
            return F.gelu(y)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_rms_smax2_gelu[(rows,)](
            x2, self.rms0_w, y,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
