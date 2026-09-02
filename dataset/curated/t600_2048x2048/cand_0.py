import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 600
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _bias_softmax_kernel(
    Y_ptr, B_ptr, Out_ptr,
    N, stride_ym, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # load matmul output row (fp16) and bias (fp16)
    y = tl.load(Y_ptr + row * stride_ym + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # bias add in fp16 (matches reference x + b1 in fp16)
    y = (y + b).to(tl.float16)

    # softmax with fp32 accumulation (matches PyTorch fp16 softmax)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float("-inf"))
    row_max = tl.max(yf, axis=0)
    yf = yf - row_max
    num = tl.math.exp(yf)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = x @ self.W0
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_softmax_kernel[(m,)](
            y, self.b1, out,
            n, y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=16,
        )
        return out
