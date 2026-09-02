import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 388
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _gelu_scale_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact GELU computed in fp32 (matches PyTorch opmath), rounded to fp16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # scale in fp32 then round to fp16 (matches PyTorch half*scalar semantics)
    s = (g * SCALE).to(tl.float16).to(tl.float32)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    s = tl.where(mask, s, float("-inf"))
    row_max = tl.max(s, axis=0)
    e = tl.exp(s - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y_ptr + row * stride_y + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores on A100)
        h = x @ self.W0

        Mrows, N = h.shape
        y = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 1024:
            num_warps = 8
        if BLOCK >= 4096:
            num_warps = 16

        _gelu_scale_softmax_kernel[(Mrows,)](
            h, y,
            N, h.stride(0), y.stride(0),
            SCALE=1.0332,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
