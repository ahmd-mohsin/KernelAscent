import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 776
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_row_kernel(
    X, B1, LN_G, LN_B, RMS_W, Y,
    D, stride_x, stride_y,
    EPS_LN, EPS_RMS, SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    # ---- softmax (fp32 math, fp16 rounding at boundary, matching PyTorch) ----
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, 0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, 0)
    sm = (e / denom).to(tl.float16)

    # ---- + b1 (opmath float, cast back to fp16) ----
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x1 = (sm.to(tl.float32) + b1).to(tl.float16)

    # ---- layer norm (fp32 math, fp16 out) ----
    xf = x1.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), 0) / D
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, 0) / D
    rstd = tl.math.rsqrt(var + EPS_LN)
    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    ln = (diff * rstd * g + b).to(tl.float16)

    # ---- rms norm (explicit fp32, cast to fp16, then * w in opmath float) ----
    lf = ln.to(tl.float32)
    ms = tl.sum(tl.where(mask, lf * lf, 0.0), 0) / D
    r = tl.math.rsqrt(ms + EPS_RMS)
    rms = (lf * r).to(tl.float16)
    w = tl.load(RMS_W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (rms.to(tl.float32) * w).to(tl.float16)

    # ---- * scalar (opmath float, cast fp16) ----
    y = (y.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_row_kernel[(rows,)](
            x2, self.b1, self.ln2_g, self.ln2_b, self.rms3_w, y,
            d, x2.stride(0), y.stride(0),
            1e-5, 1e-6, 1.0801,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        x = torch.softmax(x, dim=-1)
        x = x + self.b1
        x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        x = x * 1.0801
        return x
