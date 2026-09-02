import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 827
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b0_ptr, b2_ptr, out_ptr, n_cols, stride_row,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(x_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(b2_ptr + cols, mask=mask, other=0.0)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # x + b0 (fp32 math, round to fp16 to match eager)
    v = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.float16)
    # gelu (exact, erf-based)
    vf = v.to(tl.float32)
    v = (vf * 0.5 * (1.0 + tl.math.erf(vf * INV_SQRT2))).to(tl.float16)
    # + b2
    v = (v.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)
    # gelu
    vf = v.to(tl.float32)
    v = (vf * 0.5 * (1.0 + tl.math.erf(vf * INV_SQRT2))).to(tl.float16)
    # scale
    v = (v.to(tl.float32) * 1.2166).to(tl.float16)

    # softmax in fp32 (matches torch half softmax which accumulates in float)
    vf = tl.where(mask, v.to(tl.float32), float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, cols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(cols)
        _fused_kernel[(rows,)](
            x, self.b0, self.b2, out, cols, x.stride(0),
            BLOCK=BLOCK, num_warps=8,
        )
        return out
