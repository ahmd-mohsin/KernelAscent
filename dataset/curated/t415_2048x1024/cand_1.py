import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 415
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_kernel(X, Y, n_cols, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # emulate fp16 elementwise ops (round to fp16 after each step)
    x = (x * 1.0319).to(tl.float16)
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16)
    g = (g * 1.3895).to(tl.float16)

    # softmax in fp32 (matches torch internal accumulation)
    z = g.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
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
            x = x * 1.0319
            x = F.gelu(x)
            x = x * 1.3895
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        n_rows, n_cols = x.shape[-2], x.shape[-1]
        x2 = x.view(-1, n_cols)
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(x2.shape[0],)](
            x2, y, n_cols, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view_as(x)
