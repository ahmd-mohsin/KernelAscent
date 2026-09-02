import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 112
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_gelu2_scale_softmax(
    X, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # first exact GELU in fp32, round back to bf16 (match PyTorch op-by-op)
    xf = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g = g.to(tl.bfloat16)

    # second exact GELU
    gf = g.to(tl.float32)
    g2 = 0.5 * gf * (1.0 + tl.math.erf(gf * INV_SQRT2))
    g2 = g2.to(tl.bfloat16)

    # scale (fp32 compute, round to bf16)
    s = (g2.to(tl.float32) * 1.1137).to(tl.bfloat16)

    # softmax in fp32
    sf = tl.where(mask, s.to(tl.float32), float('-inf'))
    row_max = tl.max(sf, axis=0)
    e = tl.exp(sf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu2_scale_softmax[(m,)](
            h, y,
            h.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
