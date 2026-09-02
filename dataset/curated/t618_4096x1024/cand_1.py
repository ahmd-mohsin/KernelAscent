import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 618
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_scale_relu_softmax(
    X, Y, n_cols, stride_x, stride_y,
    S1: tl.constexpr, S2: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))
    # replicate bf16 rounding of the two separate multiplies
    x = (x.to(tl.float32) * S1).to(tl.bfloat16)
    x = (x.to(tl.float32) * S2).to(tl.bfloat16)
    x = tl.maximum(x, 0.0).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            x = x * 1.3747
            x = x * 1.155
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)
        x = x.contiguous()
        n_rows, n_cols = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2d = x.view(-1, n_cols)
        n_rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_scale_relu_softmax[(n_rows,)](
            x2d, y, n_cols, x2d.stride(0), y.stride(0),
            S1=1.3747, S2=1.155, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(x.shape)
