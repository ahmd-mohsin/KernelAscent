import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 56
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, out_ptr, n_cols, stride_row,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)

    # x = x * 1.1687 (computed in fp32, rounded to bf16 like PyTorch)
    x = (x * 1.1687).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32, output rounded to bf16
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    sm = (num / denom).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf) in fp32, rounded to bf16
    g = sm * 0.5 * (1.0 + tl.math.erf(sm * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # add bias
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (g + b).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_kernel[(n_rows,)](
            x, self.b4, out, n_cols, x.stride(0),
            BLOCK=BLOCK, num_warps=4,
        )
        return out
