import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 701
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _fused_bias_softmax_scale(
    X, B, Y,
    n_cols,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # match reference: fp16 add of bias, then softmax in fp32 accumulation
    z = (x + b).to(tl.float16).to(tl.float32)
    z = tl.where(mask, z, float('-inf'))

    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s) * SCALE

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 4096 else 4
        _fused_bias_softmax_scale[(m,)](
            x, self.b0, y,
            n,
            x.stride(0), y.stride(0),
            SCALE=1.2808,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
