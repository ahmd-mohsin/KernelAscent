import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 792
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))

    # emulate fp16 intermediate rounding of the two scalar multiplies
    x = (x.to(tl.float32) * 1.2833).to(tl.float16)
    x = (x.to(tl.float32) * 1.3985).to(tl.float16)

    # relu
    x = tl.maximum(x, 0.0)

    # exact gelu (erf), computed in fp32 then rounded to fp16
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # softmax in fp32
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.2833
            x = x * 1.3985
            x = torch.relu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        n_rows, n_cols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(n_rows,)](
            x, y, n_cols, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
