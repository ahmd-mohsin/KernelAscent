import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 225
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _softmax_bias_relu_kernel(
    X_ptr, B_ptr, Out_ptr,
    N,
    stride_xm,
    stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    # match reference numerics: softmax computed in fp16 by torch on fp16 input,
    # then add fp16 bias, relu. torch computes softmax in fp32 internally and
    # casts back to fp16, so cast here.
    sm16 = sm.to(tl.float16)

    b = tl.load(B_ptr + cols, mask=mask, other=0.0)
    y = sm16 + b
    y = tl.maximum(y, 0.0)

    tl.store(Out_ptr + row * stride_om + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS handles the GEMM (tensor cores on A100)
        y = x @ self.W0
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 1024 else 4
        _softmax_bias_relu_kernel[(Mrows,)](
            y, self.b2, out,
            N,
            y.stride(0),
            out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
