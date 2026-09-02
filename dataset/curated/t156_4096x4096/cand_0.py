import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 156
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _fused_epilogue_softmax(
    X, B3, B4, Y,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # x = x * 1.3741  (computed in fp32, rounded to fp16, matching PyTorch half TensorIterator)
    x = (x.to(tl.float32) * 1.3741).to(tl.float16)
    # relu
    x = tl.maximum(x, 0.0)
    # + b3 (fp32 compute, fp16 round)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    x = (x.to(tl.float32) + b3.to(tl.float32)).to(tl.float16)
    # + b4
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    x = (x.to(tl.float32) + b4.to(tl.float32)).to(tl.float16)

    # softmax in fp32 (matching PyTorch's half softmax which accumulates in float)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.math.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_y + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)

        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_epilogue_softmax[(m,)](
            h, self.b3, self.b4, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
