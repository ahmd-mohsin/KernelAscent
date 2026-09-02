import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 939
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _bias_softmax_kernel(
    X, B, Out,
    N, stride_xm, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in bf16 (rounded), matching reference elementwise add
    v = (x + b).to(tl.bfloat16)

    # softmax with fp32 accumulation
    vf = v.to(tl.float32)
    vf = tl.where(mask, vf, float('-inf'))
    m = tl.max(vf, axis=0)
    e = tl.exp(vf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out + row * stride_om + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _bias_softmax_kernel[(Mrows,)](
            y, self.b1, out,
            N, y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
