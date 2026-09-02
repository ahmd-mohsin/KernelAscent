import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 136
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, out_ptr, n_cols, stride_x, stride_o,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # x * 1.0831 (rounded to bf16), then * 1.4976 (rounded to bf16)
    x = (x.to(tl.float32) * 1.0831).to(tl.bfloat16)
    x = (x.to(tl.float32) * 1.4976).to(tl.bfloat16)
    # x + b2 (bf16 add computed in fp32, rounded to bf16)
    x = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    num = tl.exp(xf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    sm = (num / denom).to(tl.bfloat16)

    # gelu (erf variant) in fp32 on bf16 values
    g = sm.to(tl.float32)
    out = 0.5 * g * (1.0 + tl.erf(g * 0.7071067811865476))
    out = out.to(tl.bfloat16)

    tl.store(out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        n_rows, n_cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(n_rows,)](
            x, self.b2, out, n_cols,
            x.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
