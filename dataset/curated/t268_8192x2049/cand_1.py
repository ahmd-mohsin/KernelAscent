import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 268
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _rmsnorm_scale_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + EPS)

    # match: (xf * rsqrt).to(bf16) * w  -> bf16 (fp32 opmath, round to bf16), then * scale -> bf16
    norm_bf16 = (xf * rstd).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y1 = (norm_bf16.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    y2 = (y1.to(tl.float32) * SCALE).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, y2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.dim() == 2
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _rmsnorm_scale_kernel[(m,)](
            x, self.rms0_w, y,
            x.stride(0), y.stride(0),
            N=n, EPS=1e-6, SCALE=1.4423,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
