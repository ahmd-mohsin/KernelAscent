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
    X, B, Out,
    stride_xm, stride_om,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in bf16 rounding to match reference, then relu
    t = (x + b).to(tl.bfloat16)
    t = tl.maximum(t, 0.0)

    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))

    m = tl.max(tf, axis=0)
    e = tl.exp(tf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Out + row * stride_om + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _bias_relu_softmax_kernel[(m,)](
            y, self.b1, out,
            y.stride(0), out.stride(0),
            n, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
