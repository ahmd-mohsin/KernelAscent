import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 792
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_kernel(X, Y, D_dim, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_dim

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))
    xf = x.to(tl.float32)

    # x = x * 1.2833  (fp16 tensor op: fp32 compute, round to fp16)
    xf = (xf * 1.2833).to(tl.float16).to(tl.float32)
    # x = x * 1.3985
    xf = (xf * 1.3985).to(tl.float16).to(tl.float32)
    # relu (exact, no rounding change)
    xf = tl.maximum(xf, 0.0)
    # gelu (erf form), fp32 compute then round to fp16
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * inv_sqrt2))
    g = g.to(tl.float16).to(tl.float32)

    # softmax in fp32
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = x * 1.2833
            x = x * 1.3985
            x = torch.relu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x, y, d, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
