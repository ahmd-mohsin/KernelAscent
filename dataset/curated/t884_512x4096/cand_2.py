import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 884
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _bias_relu_softmax_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # bias add + relu in bf16 (matches reference: bf16 + bf16 -> bf16, relu bf16)
    x = x + b
    zero = tl.zeros_like(x)
    x = tl.maximum(x, zero)

    # softmax in fp32 (matches PyTorch internal fp32 accumulation for bf16)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    xf = xf - row_max
    num = tl.exp(xf)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(X_ptr.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = torch.mm(x, self.W0)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_relu_softmax_kernel[(m,)](
            y, self.b1, out,
            n, y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
