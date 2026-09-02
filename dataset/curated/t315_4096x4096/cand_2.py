import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 315
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _softmax_bias_kernel(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)
    row_max = tl.max(x, axis=0)
    num = tl.exp(x - row_max)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # Match reference: softmax computed in fp32 internally by torch, cast to bf16, then add bias in bf16
    sm_bf16 = sm.to(tl.bfloat16)
    out = sm_bf16 + b.to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _softmax_bias_kernel[(Mrows,)](
            y, self.b2, out,
            N, y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
