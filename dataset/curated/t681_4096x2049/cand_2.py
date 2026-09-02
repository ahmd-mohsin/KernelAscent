import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 681
M, D, DT = 4096, 2049, torch.bfloat16


@triton.jit
def _bias_softmax_scale_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_xm, stride_om,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # bias add in bf16 (matches reference: x + b1 in bf16)
    z = (x + b).to(tl.bfloat16)

    # softmax with fp32 accumulation (matches PyTorch bf16 softmax)
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float("-inf"))
    row_max = tl.max(zf, axis=0)
    num = tl.exp(zf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    sm = (num / denom).to(tl.bfloat16)

    # relu(relu(softmax)) == softmax (nonnegative); scale in fp32 opmath
    out = (sm.to(tl.float32) * SCALE).to(tl.bfloat16)
    tl.store(Out_ptr + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_softmax_scale_kernel[(m,)](
            y, self.b1, out,
            n, y.stride(0), out.stride(0),
            SCALE=1.1457,
            BLOCK_N=BLOCK_N,
            num_warps=16,
        )
        return out
