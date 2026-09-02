import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 492
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_relu_gelu_softmax(X, Y, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X + row * stride + offs

    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # exact GELU (erf-based), computed in fp32 then rounded to bf16
    # to match the reference's bf16 intermediate
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch's bf16 softmax)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, 0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = e / s

    tl.store(Y + row * stride + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores on bf16)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_relu_gelu_softmax[(m,)](
            h, out, n, h.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
