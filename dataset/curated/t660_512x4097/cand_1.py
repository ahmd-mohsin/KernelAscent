import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 660
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _softmax_gelu_kernel(
    x_ptr, out_ptr,
    n_cols,
    stride_x, stride_o,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (numerically stable, fp32 accumulation like PyTorch's half softmax)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom

    # cast to fp16 (softmax output dtype), then gelu computed in fp32
    s16 = s.to(tl.float16)
    v = s16.to(tl.float32)
    g = 0.5 * v * (1.0 + tl.math.erf(v * 0.70710678118654752440))

    tl.store(out_ptr + row * stride_o + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W2 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        num_warps = 16 if BLOCK_SIZE >= 8192 else 8
        _softmax_gelu_kernel[(n_rows,)](
            x, out, n_cols,
            x.stride(0), out.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return out @ self.W2
