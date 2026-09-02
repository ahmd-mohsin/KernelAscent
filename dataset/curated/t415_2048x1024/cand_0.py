import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 415
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # x = x * 1.0319 (emulate fp16 rounding between ops like PyTorch)
    x = (x * 1.0319).to(tl.float16).to(tl.float32)

    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # x = x * 1.3895
    g = (g * 1.3895).to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch half softmax)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.0319
            x = F.gelu(x)
            x = x * 1.3895
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2 = x.view(-1, n_cols)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_kernel[(n_rows,)](
            x2, y, n_cols,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
