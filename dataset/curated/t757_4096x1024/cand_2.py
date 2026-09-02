import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 757
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # scale (match bf16 rounding of x * 1.487)
    x = (x.to(tl.float32) * 1.487).to(tl.bfloat16)
    # relu
    x = tl.maximum(x, 0.0)
    # gelu (exact, erf) in fp32, round to bf16
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = xf.to(tl.bfloat16)
    # gelu again
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = xf.to(tl.bfloat16)

    # softmax in fp32 (as PyTorch does for bf16)
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
        if not x.is_cuda or x.dtype != torch.bfloat16:
            x = x * 1.487
            x = torch.relu(x)
            x = F.gelu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(n_rows,)](
            x2, y, n_cols, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
