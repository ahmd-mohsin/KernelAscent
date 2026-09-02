import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 265
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_relu_scale_softmax(
    X_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)
    # relu (exact in bf16)
    xf = x.to(tl.float32)
    xf = tl.maximum(xf, 0.0)
    # scalar mul in fp32 opmath, rounded back to bf16 (matches torch semantics)
    xf = (xf * S1).to(tl.bfloat16).to(tl.float32)
    # second relu is a no-op on non-negative values, kept implicitly
    xf = (xf * S2).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 accumulation (matches torch bf16 softmax opmath)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (same as reference)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_relu_scale_softmax[(Mrows,)](
            h, out,
            N,
            h.stride(0), out.stride(0),
            1.3853, 1.1094,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
