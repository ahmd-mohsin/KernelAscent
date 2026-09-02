import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 206
M, D, DT = 512, 512, torch.float16

_SCALE = 1.1679 * 1.3251


@triton.jit
def _fused_scale_gelu_softmax(
    X, Y, N, stride_x, stride_y, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    x = x * scale
    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.1679
            x = x * 1.3251
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)
        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        Mrows, N = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16
        _fused_scale_gelu_softmax[(Mrows,)](
            x2, y, N, x2.stride(0), y.stride(0), _SCALE,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
