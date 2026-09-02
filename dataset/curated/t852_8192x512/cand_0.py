import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 852
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_bias_act_softmax(
    Y_ptr, B1_ptr, B5_ptr, OUT_ptr,
    N, stride_y, stride_o,
    S1, S2,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    y = tl.load(Y_ptr + row * stride_y + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b5 = tl.load(B5_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b1  (bf16 rounding as in eager op)
    t = (y + b1).to(tl.bfloat16).to(tl.float32)
    # relu (exact, no rounding needed)
    t = tl.maximum(t, 0.0)
    # x = x * 1.4375  (round to bf16)
    t = (t * S1).to(tl.bfloat16).to(tl.float32)
    # x = x * 1.0728  (round to bf16)
    t = (t * S2).to(tl.bfloat16).to(tl.float32)
    # x = x + b5  (round to bf16)
    t = (t + b5).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (as PyTorch does for bf16 inputs)
    t = tl.where(mask, t, float("-inf"))
    row_max = tl.max(t, axis=0)
    e = tl.math.exp(t - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(OUT_ptr + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 tensor-core matmul
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_bias_act_softmax[(Mrows,)](
            y, self.b1, self.b5, out,
            N, y.stride(0), out.stride(0),
            1.4375, 1.0728,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
