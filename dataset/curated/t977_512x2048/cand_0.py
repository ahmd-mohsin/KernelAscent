import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 977
M, D, DT = 512, 2048, torch.float16


@triton.jit
def fused_relu_bias_relu_softmax(
    X, B, Y,
    N_COLS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N_COLS

    x = tl.load(X + row * N_COLS + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # relu(x) in fp16, add bias in fp16, relu again (match reference dtype semantics)
    x = tl.maximum(x, 0.0)
    x = x + b
    x = tl.maximum(x, 0.0)

    # softmax with fp32 accumulation (matches PyTorch's half softmax)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float("-inf"))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y + row * N_COLS + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        fused_relu_bias_relu_softmax[(m,)](
            x, self.b1, y, n, BLOCK=BLOCK, num_warps=num_warps
        )
        return y
