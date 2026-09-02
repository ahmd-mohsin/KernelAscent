import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 827
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, b2_ptr, out_ptr,
    n_cols,
    stride_xm, stride_om,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # x = x + b0 (fp16 rounding to match reference)
    x = (x + b0).to(tl.float16)

    # gelu (exact, erf-based) computed in fp32, rounded back to fp16
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # x = x + b2
    x = (x + b2).to(tl.float16)

    # gelu again
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # scale
    x = (x * 1.2166).to(tl.float16)

    # softmax over the row (fp32 accumulation, like torch)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(out_ptr + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            x = F.gelu(x)
            x = x + self.b2
            x = F.gelu(x)
            x = x * 1.2166
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x, self.b0, self.b2, out,
            n,
            x.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
