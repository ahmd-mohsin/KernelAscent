import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 390
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _double_softmax_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # first softmax (fp32 compute, like PyTorch's softmax on bf16 input)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, axis=0)
    p1 = e1 / s1

    # match reference: intermediate is rounded to bf16 between the two softmaxes
    p1 = p1.to(tl.bfloat16).to(tl.float32)
    p1 = tl.where(mask, p1, float('-inf'))

    # second softmax
    m2 = tl.max(p1, axis=0)
    e2 = tl.exp(p1 - m2)
    s2 = tl.sum(e2, axis=0)
    p2 = e2 / s2

    tl.store(Y + row * stride_ym + cols, p2.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _double_softmax_kernel[(m,)](
            x, y, n,
            x.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
