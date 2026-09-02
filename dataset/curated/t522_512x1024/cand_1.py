import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 522
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_rms_scale_ln_kernel(
    X, W, G, B, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D
    base = row * D

    # ---- RMSNorm (fp32 math, round to fp16, then fp16 weight mult) ----
    x = tl.load(X + base + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    inv_rms = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (x * inv_rms).to(tl.float16)              # .to(x.dtype)
    w = tl.load(W + cols, mask=mask, other=0.0)    # fp16
    y = xh * w                                     # fp16 multiply

    # ---- scale by 1.3348 (opmath fp32, result rounded to fp16) ----
    y = (y.to(tl.float32) * 1.3348).to(tl.float16)

    # ---- LayerNorm (fp32 accumulation, fp16 output) ----
    yf = y.to(tl.float32)
    mean = tl.sum(tl.where(mask, yf, 0.0), axis=0) / D
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = ((yf - mean) * rstd * g + b).to(tl.float16)

    tl.store(Y + base + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x * 1.3348
            return F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_rms_scale_ln_kernel[(rows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, y,
            D=d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
