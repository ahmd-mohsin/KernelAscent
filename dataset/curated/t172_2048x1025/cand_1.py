import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 172
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _fused_gelu_scale_softmax_kernel(
    x_ptr, out_ptr,
    n_cols,
    stride_row,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact GELU in fp32, then round to bf16 (matches PyTorch bf16 gelu)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale in fp32, round to bf16 (matches PyTorch scalar mul on bf16)
    s = (g * SCALE).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    s = tl.where(mask, s, float('-inf'))
    m = tl.max(s, axis=0)
    e = tl.exp(s - m)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(out_ptr + row * stride_row + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x * 1.2863
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        n_rows, n_cols = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, n_cols)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 4
        if BLOCK_SIZE >= 2048:
            num_warps = 8
        if BLOCK_SIZE >= 8192:
            num_warps = 16

        _fused_gelu_scale_softmax_kernel[(rows,)](
            x2, out,
            n_cols,
            x2.stride(0),
            SCALE=1.2863,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out.view(x.shape)
