import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 172
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _gelu_scale_softmax_kernel(
    X, Y,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU in fp32, rounded to bf16 to match elementwise op semantics
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale, rounded to bf16 (matches bf16 tensor * python scalar)
    s = (g * SCALE).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches CUDA softmax accumulation dtype)
    s = tl.where(mask, s, float('-inf'))
    row_max = tl.max(s, axis=0)
    e = tl.exp(s - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            y = F.gelu(x)
            y = y * 1.2863
            return torch.softmax(y, dim=-1)

        x = x.contiguous()
        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.view(-1, n)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _gelu_scale_softmax_kernel[(m,)](
            x2, y,
            n, x2.stride(0), y.stride(0),
            SCALE=1.2863,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
