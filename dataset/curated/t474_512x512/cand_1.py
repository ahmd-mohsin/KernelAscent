import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 474
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, out_ptr, n_cols, stride_x, stride_o,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # relu, scale, bias
    x = tl.maximum(x, 0.0)
    x = x * 1.0381
    x = x + b

    # exact gelu
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))

    # softmax
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(out_ptr + row * stride_o + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        n_rows, n_cols = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        _fused_kernel[(n_rows,)](
            x2, self.b2, out, n_cols,
            x2.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4 if BLOCK <= 1024 else 8,
        )
        return out.view(orig_shape)
