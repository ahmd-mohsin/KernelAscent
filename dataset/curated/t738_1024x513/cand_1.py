import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 738
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))

    # relu (in input dtype)
    x = tl.maximum(x, 0.0)

    # gelu (exact, erf) computed in fp32 like PyTorch's opmath for half
    xf = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))

    # cast back to fp16 (intermediate tensor storage), then relu
    g = g.to(tl.float16)
    g = tl.maximum(g, 0.0)

    # softmax in fp32 (matches PyTorch fp16 softmax accumulation)
    gf = tl.where(mask, g.to(tl.float32), float('-inf'))
    m = tl.max(gf, axis=0)
    e = tl.exp(gf - m)
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
            x = torch.relu(x)
            x = F.gelu(x)
            x = torch.relu(x)
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
