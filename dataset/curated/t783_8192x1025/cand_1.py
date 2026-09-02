import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 783
M, D, DT = 8192, 1025, torch.bfloat16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    # x = x * 1.2751  (round to bf16 as PyTorch would)
    x = (x * 1.2751).to(tl.bfloat16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu (exact, erf-based), computed in fp32, stored bf16
    x = (0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    x = (0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # x = x * 1.1314
    x = (x * 1.1314).to(tl.bfloat16).to(tl.float32)

    # softmax over the row (fp32 accumulation like PyTorch)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.2751
            x = F.gelu(x)
            x = F.gelu(x)
            x = x * 1.1314
            return torch.softmax(x, dim=-1)

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
        _fused_kernel[(n_rows,)](
            x2, y, n_cols, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
