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
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x_ptr = X + row * stride_row + offs
    xf = tl.load(x_ptr, mask=mask, other=float('-inf')).to(tl.float32)

    # ---- softmax 1 (fp32 math, fp16 output) ----
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm1 = (e / s).to(tl.float16)

    # ---- RMSNorm 1 ----
    f1 = sm1.to(tl.float32)
    f1 = tl.where(mask, f1, 0.0)
    ms1 = tl.sum(f1 * f1, axis=0) / D
    r1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    h1 = (f1 * r1).to(tl.float16) * w1  # fp16 multiply

    # ---- softmax 2 ----
    hf = tl.where(mask, h1.to(tl.float32), float('-inf'))
    m2 = tl.max(hf, axis=0)
    e2 = tl.exp(hf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    sm2 = (e2 / s2).to(tl.float16)

    # ---- GELU (exact, fp32 opmath) ----
    gf = sm2.to(tl.float32)
    ge = (gf * 0.5 * (1.0 + tl.math.erf(gf * 0.7071067811865476))).to(tl.float16)

    # ---- RMSNorm 4 ----
    f4 = ge.to(tl.float32)
    f4 = tl.where(mask, f4, 0.0)
    ms4 = tl.sum(f4 * f4, axis=0) / D
    r4 = 1.0 / tl.sqrt(ms4 + 1e-6)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0)
    out = (f4 * r4).to(tl.float16) * w4

    tl.store(Y + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            # fallback to reference path
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        x = x.contiguous()
        d = x.shape[-1]
        rows = x.numel() // d
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_row_kernel[(rows,)](
            x, self.rms1_w, self.rms4_w, y,
            d, d,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y
