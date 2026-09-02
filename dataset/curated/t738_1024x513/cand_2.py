import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 738
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_kernel(x_ptr, out_ptr, n_cols, stride_x, stride_o, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float('-inf'))

    # relu (in input dtype, exact)
    x = tl.where(x > 0, x, 0.0)

    # gelu (exact erf variant), computed in fp32 then rounded to fp16
    # to match PyTorch's opmath behavior for half precision
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # relu (no-op mathematically since g >= 0, kept for fidelity)
    g = tl.where(g > 0, g, 0.0)

    # softmax in fp32 (matches PyTorch internal accumulation)
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(out_ptr + row * stride_o + cols, y.to(tl.float16), mask=mask)


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
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_kernel[(n_rows,)](
            x2, out, n_cols, x2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
