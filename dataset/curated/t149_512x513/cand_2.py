import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 149
M, D, DT = 512, 513, torch.float16


@triton.jit
def _gelu_softmax_scale_kernel(
    X, Out,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), computed in fp32 then rounded to fp16
    # to match PyTorch's half-precision gelu output
    inv_sqrt2 = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))
    g = g.to(tl.float16).to(tl.float32)

    # softmax in fp32 (matches PyTorch internal float accumulation for half)
    g_m = tl.where(mask, g, float("-inf"))
    row_max = tl.max(g_m, axis=0)
    e = tl.exp(g_m - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # round softmax result to fp16, then scale (float compute, round to fp16)
    sm_h = sm.to(tl.float16).to(tl.float32)
    out = (sm_h * SCALE).to(tl.float16)

    tl.store(Out + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 matmul
        y = x @ self.W0
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _gelu_softmax_scale_kernel[(m,)](
            y, out,
            n, y.stride(0), out.stride(0),
            SCALE=1.2464,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
