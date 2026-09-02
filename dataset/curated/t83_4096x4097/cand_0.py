import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 83
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _softmax_bias_kernel(
    X, B, Y,
    N,
    stride_xm,
    stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    row_max = tl.max(x, axis=0)
    x = x - row_max
    num = tl.exp(x)
    denom = tl.sum(num, axis=0)
    sm = num / denom

    # round softmax output to bf16 (matches the intermediate tensor in eager mode)
    sm_bf16 = sm.to(tl.bfloat16)

    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (sm_bf16.to(tl.float32) + b).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (tensor cores on A100)
        h = x @ self.W0

        m, n = h.shape
        y = torch.empty_like(h)

        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16

        _softmax_bias_kernel[(m,)](
            h, self.b2, y,
            n,
            h.stride(0),
            y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
