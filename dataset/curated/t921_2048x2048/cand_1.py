import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 921
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _bias_gelu_softmax_kernel(X, B, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in fp16 (matches reference x + b1 on half tensors)
    t = (x + b).to(tl.float32)

    # exact GELU (erf), computed in fp32 like PyTorch's opmath for half
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))
    # cast back to fp16 (reference materializes gelu output in half), then fp32 for softmax
    g = g.to(tl.float16).to(tl.float32)

    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * N + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 tensor-core GEMM, identical to reference
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _bias_gelu_softmax_kernel[(Mrows,)](
            h, self.b1, out, N,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
