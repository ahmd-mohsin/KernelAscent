import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 82
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(
    X_ptr, Y_ptr,
    N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf'))
    x = x.to(tl.float32)
    # relu
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))
    # softmax (numerically stable, fp32 accumulation like PyTorch)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom
    tl.store(Y_ptr + row * stride_ym + cols, out.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # scale in same dtype (matches reference rounding), then cuBLAS matmul
        x = x * 1.3815
        y = x @ self.W1

        M_, N_ = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N_)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _relu_softmax_kernel[(M_,)](
            y, out,
            N_,
            y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
