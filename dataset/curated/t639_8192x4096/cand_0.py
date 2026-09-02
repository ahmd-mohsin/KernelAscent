import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 639
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _fused_kernel(X, Y, stride_x, stride_y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_
    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)

    # relu (fp16)
    x = tl.maximum(x, 0.0)
    # * 1.3697, rounded to fp16 like PyTorch
    x = (x.to(tl.float32) * 1.3697).to(tl.float16)
    # exact gelu: x * 0.5 * (1 + erf(x / sqrt(2))), rounded to fp16
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = g.to(tl.float16)
    # * 1.1166, rounded to fp16
    x = (x.to(tl.float32) * 1.1166).to(tl.float16)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)
    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not (x.is_cuda and x.dtype == torch.float16 and x.dim() == 2):
            x = torch.relu(x)
            x = x * 1.3697
            x = F.gelu(x)
            x = x * 1.1166
            return torch.softmax(x, dim=-1)
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](x, y, x.stride(0), y.stride(0), d, BLOCK,
                            num_warps=num_warps)
        return y
