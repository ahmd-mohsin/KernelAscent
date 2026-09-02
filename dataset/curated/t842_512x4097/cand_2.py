import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 842
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_bias_rms_scale_kernel(
    X, B0, W, OUT,
    D_dim,
    stride_xm,
    stride_om,
    eps,
    scale1, scale2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_dim

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B0 + cols, mask=mask, other=0.0)

    # x + b0 : fp16 inputs, computed in fp32 opmath, rounded to fp16
    xh = (x.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # RMS norm in fp32
    xf = xh.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / D_dim
    inv = 1.0 / tl.sqrt(ms + eps)
    n = (xf * inv).to(tl.float16)

    # * rms1_w (fp16 op, fp32 opmath, round to fp16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (n.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # * 1.3737, then * 1.2539 (each rounds to fp16)
    y = (y.to(tl.float32) * scale1).to(tl.float16)
    y = (y.to(tl.float32) * scale2).to(tl.float16)

    tl.store(OUT + row * stride_om + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 16 if BLOCK >= 8192 else 8
        _fused_bias_rms_scale_kernel[(m,)](
            x, self.b0, self.rms1_w, out,
            d,
            x.stride(0),
            out.stride(0),
            1e-6,
            1.3737, 1.2539,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
