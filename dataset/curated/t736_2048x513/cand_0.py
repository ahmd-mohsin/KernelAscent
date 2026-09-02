import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 736
M, D, DT = 2048, 513, torch.bfloat16


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
    # ReLU
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))
    # Softmax (numerically stable, fp32 accumulation like PyTorch)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_ym + cols, y.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 512, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _relu_softmax_kernel[(m,)](
            h, out, n,
            h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
