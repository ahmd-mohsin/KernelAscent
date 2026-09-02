import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 955
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)
    # fused relu
    x = tl.maximum(x, 0.0)
    # mask out-of-bounds for softmax reduction
    x = tl.where(mask, x, float('-inf'))

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    y = num / denom

    tl.store(Y + row * stride_ym + offs, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4097, 512, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)

    def forward(self, x):
        # ReLU (elementwise, cheap) then cuBLAS matmul (best for GEMM on A100)
        x = torch.relu(x)
        h = x @ self.W1  # (M, 512), bf16

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK_N = triton.next_power_of_2(N)
        grid = (Mrows,)
        _relu_softmax_kernel[grid](
            h, out,
            N,
            h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
