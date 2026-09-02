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
    stride_xm, stride_ym,
    N: tl.constexpr,
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
    sm = num / den

    # round softmax result to bf16 (matches torch.softmax output dtype)
    sm_bf16 = sm.to(tl.bfloat16)

    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = sm_bf16.to(tl.float32) + b

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM with fp32 accumulate
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _softmax_bias_kernel[(m,)](
            h, self.b2, y,
            h.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
