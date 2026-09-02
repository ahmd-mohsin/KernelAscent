import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 873
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_gelu_softmax_scale_relu(
    X, Y, N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))
    xf = x.to(tl.float32)

    # exact (erf-based) GELU in fp32, then round to bf16 (matches PyTorch opmath)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = tl.where(mask, g, float('-inf'))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32, round to bf16
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.bfloat16).to(tl.float32)

    # scale in fp32, round to bf16, relu
    out = (sm * SCALE).to(tl.bfloat16)
    out = tl.maximum(out, 0.0)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        y = torch.empty_like(h)
        m, n = h.shape
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_softmax_scale_relu[(m,)](
            h, y, n, h.stride(0), y.stride(0),
            SCALE=1.1536, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
