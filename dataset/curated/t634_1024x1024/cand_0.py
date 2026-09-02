import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 634
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 accumulation, as PyTorch does for fp16 inputs)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = e1 / s1
    # cast to fp16 (intermediate tensor dtype in reference)
    p1 = p1.to(tl.float16).to(tl.float32)

    # scale (opmath fp32, output fp16)
    y = (p1 * 1.4236).to(tl.float16).to(tl.float32)
    y = tl.where(mask, y, float('-inf'))

    # softmax 2
    m2 = tl.max(y, axis=0)
    e2 = tl.exp(y - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2
    p2 = p2.to(tl.float16).to(tl.float32)

    # gelu (erf-based, fp32 opmath) then cast fp16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * p2 * (1.0 + tl.math.erf(p2 * INV_SQRT2))
    g = g.to(tl.float16)

    # relu
    zero = tl.zeros_like(g)
    out = tl.maximum(g, zero)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = y * 1.4236
            y = torch.softmax(y, dim=-1)
            y = F.gelu(y)
            return torch.relu(y)

        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2d.shape
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_kernel[(n_rows,)](
            x2d, out, n_cols,
            x2d.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
