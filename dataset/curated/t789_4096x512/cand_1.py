import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 789
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_scale_relu_softmax(
    X, Y, n_cols,
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # scale in fp32, round back to bf16 (matches PyTorch elementwise mul on bf16)
    x = (x.to(tl.float32) * SCALE).to(tl.bfloat16)
    # relu (idempotent, once suffices)
    zero = tl.zeros_like(x)
    x = tl.maximum(x, zero)
    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)
    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.3032
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_scale_relu_softmax[(n_rows,)](
            x2, y, n_cols,
            x2.stride(0), y.stride(0),
            SCALE=1.3032,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
