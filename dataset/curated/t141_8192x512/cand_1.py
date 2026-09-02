import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 141
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, out_ptr, n_cols, stride_x, stride_o,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # relu in bf16
    x = tl.maximum(x, 0.0)
    # add in bf16 (match reference rounding)
    y = (x + b).to(x_ptr.dtype.element_ty)
    # second relu
    z = tl.maximum(y, 0.0).to(tl.float32)

    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(out_ptr + row * stride_o + cols,
             out.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, cols = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2d = x.view(-1, cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(n_rows,)](
            x2d, self.b1, out, cols,
            x2d.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view_as(x)
