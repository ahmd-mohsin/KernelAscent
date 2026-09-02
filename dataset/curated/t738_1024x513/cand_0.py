import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 738
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_relu_gelu_relu_softmax(
    X, Y,
    n_cols,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)
    # exact gelu (erf) computed in fp32, rounded back to fp16 to match reference
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865475))
    g = g.to(tl.float16).to(tl.float32)
    # relu
    g = tl.maximum(g, 0.0)

    # softmax with fp32 accumulation
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_y + offs, out.to(tl.float16), mask=mask)


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
        x2d = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2d.shape
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_relu_gelu_relu_softmax[(n_rows,)](
            x2d, y,
            n_cols,
            x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
