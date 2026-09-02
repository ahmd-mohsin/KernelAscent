import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 381
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_scale_softmax_kernel(
    X, Y,
    n_cols,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))
    # replicate fp16 rounding of the scale multiply, then softmax in fp32
    x = (x.to(tl.float32) * SCALE).to(tl.float16).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom
    # relu is a no-op on softmax output (all values >= 0)
    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not (x.is_cuda and x.dtype == torch.float16 and x.dim() == 2):
            x = x * 1.4408
            x = torch.softmax(x, dim=-1)
            return torch.relu(x)

        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_scale_softmax_kernel[(m,)](
            x, y, n,
            x.stride(0), y.stride(0),
            SCALE=1.4408,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
