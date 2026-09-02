import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 56
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, B, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # x * 1.1687 (round to bf16 like PyTorch elementwise op output)
    x = (x * 1.1687).to(tl.bfloat16).to(tl.float32)
    x = tl.where(mask, x, float('-inf'))

    # softmax (fp32 accumulate, bf16 output)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # exact gelu (erf-based), round to bf16
    g = (p * 0.5 * (1.0 + tl.math.erf(p * 0.7071067811865476))).to(tl.bfloat16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # + bias
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (g + b).to(tl.bfloat16)
    tl.store(Y + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_kernel[(Mrows,)](x, self.b4, y, N, BLOCK=BLOCK, num_warps=4)
        return y
