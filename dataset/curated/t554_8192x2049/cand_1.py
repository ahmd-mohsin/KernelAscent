import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 554
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _softmax_relu_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    denom = tl.sum(num, axis=0)
    y = num / denom
    # relu(relu(softmax)) == softmax since softmax >= 0; still clamp for exactness
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * stride_ym + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores on A100 for bf16)
        z = x @ self.W0
        M_, N_ = z.shape
        out = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK_N >= 1024 else 4
        _softmax_relu_kernel[(M_,)](
            z, out,
            z.stride(0), out.stride(0),
            N_,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
