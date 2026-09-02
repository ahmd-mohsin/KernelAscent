import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 966
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # relu(x)
    xf = tl.maximum(x.to(tl.float32), 0.0)
    # x * scale (compute in fp32, round to bf16 to match eager intermediate)
    xf = (xf * SCALE).to(tl.bfloat16).to(tl.float32)
    # x + b3 (compute in fp32, round to bf16)
    xf = (xf + b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    # relu
    xf = tl.maximum(xf, 0.0)

    # softmax in fp32
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_act_softmax_kernel[(m,)](
            h, self.b3, y,
            n, h.stride(0), y.stride(0),
            SCALE=1.1429,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
