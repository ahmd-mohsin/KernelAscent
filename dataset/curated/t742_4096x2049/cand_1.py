import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 742
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _bias_relu_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load matmul output (fp16) and bias (fp16); add + relu in fp16
    # to exactly match the reference (x + b1 -> relu happen in fp16).
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    x = x + b
    zero = tl.zeros(x.shape, dtype=x.dtype)
    x = tl.maximum(x, zero)  # relu in fp16

    # Softmax in fp32 (matches PyTorch's fp16 softmax which accumulates in fp32)
    xf = tl.where(mask, x.to(tl.float32), float('-inf'))
    row_max = tl.max(xf, axis=0)
    num = tl.exp(xf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y_ptr + row * stride_y + offs, out.to(Y_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS tensor cores (identical op to reference)
        y = x @ self.W0

        M_, N_ = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N_)
        grid = (M_,)
        _bias_relu_softmax_kernel[grid](
            y, self.b1, out,
            N_, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
