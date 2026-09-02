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
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # emulate bf16 add (x + b in bf16), then relu
    v = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    v = tl.maximum(v, 0.0).to(tl.float32)

    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Out + row * stride_xm + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # (M, 512) bf16, tensor-core matmul
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _bias_relu_softmax_kernel[(m,)](
            y, self.b1, out,
            y.stride(0),
            N=n,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
