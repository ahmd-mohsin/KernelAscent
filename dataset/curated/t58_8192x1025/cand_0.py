import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 58
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _relu_softmax_kernel(
    X, Y,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    # fused double relu == single relu
    x = tl.maximum(x, 0.0)
    # masked lanes must not affect max
    x = tl.where(mask, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_y + offs, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS handles the GEMM optimally on A100 (tensor cores, fp16)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()

        rows, N = y.shape
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        # fused relu + relu + softmax, in-place on the GEMM output
        _relu_softmax_kernel[(rows,)](
            y, y,
            N,
            y.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
