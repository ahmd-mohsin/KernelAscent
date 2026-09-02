import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 81
M, D, DT = 8192, 4096, torch.bfloat16


@triton.jit
def _relu_scale_softmax_kernel(
    X_ptr, Out_ptr,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)  # bf16

    # relu in bf16
    x = tl.maximum(x, 0.0)

    # x * 1.1524 (compute in fp32, round to bf16 — matches PyTorch opmath)
    xf = x.to(tl.float32) * 1.1524
    x = xf.to(tl.bfloat16)

    # x * 1.1059
    xf = x.to(tl.float32) * 1.1059
    x = xf.to(tl.bfloat16)

    # softmax in fp32 (matches PyTorch accumulate type)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _relu_scale_softmax_kernel[(Mrows,)](
            y, out,
            N, y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
