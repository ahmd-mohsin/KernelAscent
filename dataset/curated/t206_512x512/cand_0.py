import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 206
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_scale_gelu_softmax(X, Y, n_cols, stride_x, stride_y,
                              BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # scale 1: emulate fp16 rounding like the reference (half tensor ops)
    x = (x * 1.1679).to(tl.float16).to(tl.float32)
    # scale 2
    x = (x * 1.3251).to(tl.float16).to(tl.float32)

    # exact GELU (erf), computed in fp32 (PyTorch opmath), stored as fp16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax with fp32 accumulation (PyTorch acc_type for half is float)
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        assert x.is_cuda and x.dim() == 2
        x = x.contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_scale_gelu_softmax[(n_rows,)](
            x, y, n_cols, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
