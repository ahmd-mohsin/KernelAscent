import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 527
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_double_rmsnorm_relu(
    X, W0, W2, Y,
    stride_x, stride_y,
    D_: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 ----
    ms0 = tl.sum(x * x, axis=0) / D_
    inv0 = tl.math.rsqrt(ms0 + EPS)
    h = (x * inv0).to(tl.float16)  # cast to fp16 like reference

    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    h = (h.to(tl.float32) * w0.to(tl.float32)).to(tl.float16)
    # ReLU
    h = tl.maximum(h, 0.0)

    # ---- RMSNorm 2 ----
    xf = h.to(tl.float32)
    ms2 = tl.sum(xf * xf, axis=0) / D_
    inv2 = tl.math.rsqrt(ms2 + EPS)
    h2 = (xf * inv2).to(tl.float16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    h2 = (h2.to(tl.float32) * w2.to(tl.float32)).to(tl.float16)
    h2 = tl.maximum(h2, 0.0)

    tl.store(Y + row * stride_y + cols, h2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return torch.relu(x)

        x = x.contiguous()
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_double_rmsnorm_relu[(Mrows,)](
            x, self.rms0_w, self.rms2_w, y,
            x.stride(0), y.stride(0),
            D_=Dcols, EPS=1e-6, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
