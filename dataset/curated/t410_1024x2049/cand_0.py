import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 410
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _fused_gelu_relu_softmax_kernel(
    X, Y,
    n_cols,
    stride_xm, stride_ym,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then rounded to bf16 to match PyTorch
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    r = tl.maximum(g, 0.0)

    # softmax (fp32 math, masked lanes excluded)
    r = tl.where(mask, r, float('-inf'))
    row_max = tl.max(r, axis=0)
    e = tl.exp(r - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        y = torch.empty_like(x2)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK_SIZE >= 2048 else 4

        _fused_gelu_relu_softmax_kernel[(n_rows,)](
            x2, y, n_cols,
            x2.stride(0), y.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
