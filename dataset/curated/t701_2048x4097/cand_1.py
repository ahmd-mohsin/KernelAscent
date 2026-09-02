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
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # add in fp16 to match reference rounding, then upcast for softmax
    xb = (x + b).to(tl.float32)
    xb = tl.where(mask, xb, float('-inf'))

    row_max = tl.max(xb, axis=0)
    e = tl.exp(xb - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)

    y = (e / denom).to(tl.float16)
    # scale: fp16 -> fp32 multiply -> fp16 (matches PyTorch opmath)
    out = (y.to(tl.float32) * scale).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 16 if BLOCK >= 8192 else 8
        _fused_bias_softmax_scale[(n_rows,)](
            x, self.b0, y,
            n_cols,
            x.stride(0), y.stride(0),
            1.2808,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
