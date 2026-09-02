import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 966
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_epilogue_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N, SCALE,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # relu -> *scale -> +bias -> relu  (compute in bf16 to match reference numerics)
    x = tl.maximum(x, 0.0)
    x = (x.to(tl.bfloat16) * tl.full((1,), SCALE, tl.bfloat16)).to(tl.float32)
    x = (x.to(tl.bfloat16).to(tl.float32) + b)
    x = x.to(tl.bfloat16).to(tl.float32)
    x = tl.maximum(x, 0.0)

    # softmax
    x_masked = tl.where(mask, x, float("-inf"))
    m = tl.max(x_masked, axis=0)
    e = tl.exp(x_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_ym + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 tensor-core GEMM
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_epilogue_softmax[(Mrows,)](
            h, self.b3, y,
            h.stride(0), y.stride(0),
            N, 1.1429,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
