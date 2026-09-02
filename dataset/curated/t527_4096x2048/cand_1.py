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
    D_: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # --- RMSNorm 0 ---
    ms = tl.sum(xf * xf, axis=0) / D_
    r = tl.math.rsqrt(ms + 1e-6)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    h = (xf * r).to(tl.float16) * w0
    # ReLU
    h = tl.maximum(h, tl.zeros_like(h))

    # --- RMSNorm 2 ---
    hf = h.to(tl.float32)
    ms2 = tl.sum(hf * hf, axis=0) / D_
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    o = (hf * r2).to(tl.float16) * w2
    o = tl.maximum(o, tl.zeros_like(o))

    tl.store(Y + row * stride_y + cols, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda and x.dtype == torch.float16
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_double_rmsnorm_relu[(m,)](
            x, self.rms0_w, self.rms2_w, y,
            x.stride(0), y.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
