import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 341
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_relu_gelu_softmax(X, Y, stride_x, stride_y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)
    # ReLU
    x = tl.maximum(x, 0.0)
    # exact GELU (erf-based), rounded to fp16 to mirror reference elementwise op
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)
    g = tl.where(mask, g, float('-inf'))
    # softmax with fp32 accumulation (matches PyTorch half softmax behavior)
    m = tl.max(g, 0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = e / s
    tl.store(Y + row * stride_y + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        out = torch.empty_like(h)
        rows, cols = h.shape
        BLOCK = triton.next_power_of_2(cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_relu_gelu_softmax[(rows,)](
            h, out, h.stride(0), out.stride(0), cols,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
