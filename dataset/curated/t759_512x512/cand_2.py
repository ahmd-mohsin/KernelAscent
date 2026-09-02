import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 759
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _softmax_bias_relu_kernel(
    X, B, Out,
    stride_xm,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    sm = (num / den).to(tl.bfloat16)

    b = tl.load(B + cols, mask=mask, other=0.0)
    y = sm + b
    zero = tl.zeros(y.shape, dtype=y.dtype)
    y = tl.maximum(y, zero)

    tl.store(Out + row * stride_xm + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores on A100)
        y = torch.matmul(x, self.W0)

        if not y.is_cuda:
            y = torch.softmax(y, dim=-1)
            y = torch.relu(y + self.b2)
            return y

        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _softmax_bias_relu_kernel[(m,)](
            y, self.b2, out,
            y.stride(0),
            n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
