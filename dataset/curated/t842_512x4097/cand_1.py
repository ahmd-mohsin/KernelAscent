import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 842
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_bias_rmsnorm_scale(
    X, B0, W, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B0 + cols, mask=mask, other=0.0)

    # x = x + b0  (half add, correctly rounded)
    xh = (x.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # RMS in float32
    xf = xh.to(tl.float32)
    ssum = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0)
    inv = tl.math.rsqrt(ssum / N + EPS)

    # normalized -> half
    xn = (xf * inv).to(tl.float16)

    # * rms1_w (half, via f32 compute -> round)
    w = tl.load(W + cols, mask=mask, other=0.0)
    t1 = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # * 1.3737 then * 1.2539 with intermediate half rounding
    t2 = (t1.to(tl.float32) * S1).to(tl.float16)
    t3 = (t2.to(tl.float32) * S2).to(tl.float16)

    tl.store(Y + row * stride_y + cols, t3, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_rmsnorm_scale[(m,)](
            x2, self.b0, self.rms1_w, y,
            n, x2.stride(0), y.stride(0),
            EPS=1e-6, S1=1.3737, S2=1.2539,
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view(orig_shape)
