import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 339
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_gelu_bias_gelu_softmax(X, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    ptr = X + row * N + cols

    # load matmul result (fp16) and compute first GELU in fp32 (matches PyTorch opmath)
    x = tl.load(ptr).to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16)

    # bias add in fp16 (matches half + half tensor add)
    b = tl.load(B + cols)
    x = x + b

    # second GELU in fp32, round back to fp16
    xf = x.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    x = xf.to(tl.float16)

    # scalar multiply: fp32 opmath, round to fp16 (matches half * python float)
    xf = x.to(tl.float32) * 1.437
    x = xf.to(tl.float16)

    # softmax in fp32 accumulation (matches PyTorch softmax on half)
    xf = x.to(tl.float32)
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * N + cols, y.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        grid = (Mrows,)
        _fused_gelu_bias_gelu_softmax[grid](
            h, self.b2, y,
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=4,
        )
        return y
