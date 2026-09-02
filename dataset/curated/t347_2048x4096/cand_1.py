import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 347
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_epilogue_softmax(X, B2, B3, B4, Y, N, stride_x, stride_y,
                            BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)

    # relu in bf16
    x = tl.maximum(x, 0.0)
    # sequential bf16 additions (round after each, matching reference)
    x = (x + b2).to(tl.bfloat16)
    x = (x + b3).to(tl.bfloat16)
    x = (x + b4).to(tl.bfloat16)
    x = tl.maximum(x, 0.0)

    # softmax in fp32
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores)
        h = x @ self.W0
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue_softmax[(m,)](
            h, self.b2, self.b3, self.b4, y,
            n, h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
