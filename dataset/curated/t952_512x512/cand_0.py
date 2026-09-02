import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 952
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_relu_bias_softmax(X, B, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), rounded to fp16 to match PyTorch op boundary
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # relu
    r = tl.maximum(g, 0.0)

    # bias add, rounded to fp16 to match PyTorch op boundary
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    z = (r + b).to(tl.float16).to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax which accumulates in fp32)
    z = tl.where(mask, z, float('-inf'))
    m = tl.max(z, 0)
    e = tl.exp(z - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    out = (e / s).to(tl.float16)

    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_relu_bias_softmax[(rows,)](
            h, self.b3, y, N, BLOCK=BLOCK, num_warps=8
        )
        return y
