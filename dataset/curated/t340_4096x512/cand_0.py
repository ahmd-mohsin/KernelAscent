import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 340
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _softmax_scale_bias_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    den = tl.sum(num, axis=0)
    sm = num / den

    # match reference fp16 rounding: softmax -> fp16, *1.2811 -> fp16, *1.1377 -> fp16, +b -> fp16
    sm_h = sm.to(tl.float16)
    c1: tl.constexpr = 1.2811
    c2: tl.constexpr = 1.1377
    y = (sm_h * c1.to(tl.float16)).to(tl.float16)
    y = (y * c2.to(tl.float16)).to(tl.float16)

    b = tl.load(B + cols, mask=mask, other=0.0)
    y = (y + b).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS tensor-core GEMM
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _softmax_scale_bias_kernel[(m,)](
            h, self.b4, y,
            h.stride(0), y.stride(0),
            N=n, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
