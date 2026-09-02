import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 270
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _scale_softmax_kernel(X, Y, n_cols, stride_x, stride_y,
                          SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))
    # replicate: (bf16 * scalar) computed in fp32, cast back to bf16, then softmax in fp32
    xs = (x.to(tl.float32) * SCALE).to(x.dtype).to(tl.float32)
    m = tl.max(xs, axis=0)
    e = tl.exp(xs - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            return torch.softmax(x * 1.0211, dim=-1)
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _scale_softmax_kernel[(n_rows,)](
            x2, y, n_cols, x2.stride(0), y.stride(0),
            SCALE=1.0211, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
