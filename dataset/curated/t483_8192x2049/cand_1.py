import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 483
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _scale_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    n_cols,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))

    # Emulate PyTorch fp16 elementwise scaling: compute in fp32, round to fp16 each step
    x = x.to(tl.float32) * 1.1753
    x = x.to(tl.float16).to(tl.float32) * 1.0586
    x = x.to(tl.float16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))

    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = (x * 1.1753) * 1.0586
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.reshape(-1, orig_shape[-1]).contiguous()
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _scale_softmax_kernel[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
