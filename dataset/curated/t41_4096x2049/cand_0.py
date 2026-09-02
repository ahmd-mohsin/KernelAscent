import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 41
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _fused_row_kernel(
    X, W1, W4, Y,
    D: tl.constexpr,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- load input row (fp16 -> fp32) ----
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 compute, round to fp16) ----
    mx = tl.max(x, axis=0)
    e = tl.exp(x - mx)          # masked lanes: exp(-inf) = 0
    s = tl.sum(e, axis=0)
    p = e / s
    p = p.to(tl.float16).to(tl.float32)

    # ---- RMSNorm 1 (fp32 stats, round to fp16, then * w1) ----
    ms = tl.sum(p * p, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float32)
    h = (p * r).to(tl.float16).to(tl.float32) * w1
    h = h.to(tl.float16).to(tl.float32)

    # ---- softmax 2 ----
    h = tl.where(mask, h, float('-inf'))
    mx2 = tl.max(h, axis=0)
    e2 = tl.exp(h - mx2)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2
    p2 = p2.to(tl.float16).to(tl.float32)

    # ---- exact GELU (erf), round to fp16 ----
    g = p2 * 0.5 * (1.0 + tl.math.erf(p2 * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)
    g = tl.where(mask, g, 0.0)

    # ---- RMSNorm 2 ----
    ms2 = tl.sum(g * g, axis=0) / D
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((g * r2).to(tl.float16).to(tl.float32) * w4).to(tl.float16)

    tl.store(Y + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # reference fallback for CPU
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_row_kernel[(m,)](
            x2, self.rms1_w, self.rms4_w, y,
            d, x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view(orig_shape)
