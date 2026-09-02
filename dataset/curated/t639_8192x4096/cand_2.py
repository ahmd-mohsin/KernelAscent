import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 639
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    # scale
    x = x * 1.3697
    # exact gelu: x * 0.5 * (1 + erf(x / sqrt(2)))
    x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # scale
    x = x * 1.1166

    # softmax (masked positions were -inf -> after relu chain they'd be wrong,
    # so re-mask before softmax)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_y + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            y = torch.relu(x)
            y = y * 1.3697
            y = F.gelu(y)
            y = y * 1.1166
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        if BLOCK >= 8192:
            num_warps = 16
        _fused_kernel[(m,)](
            x2, y, n, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
