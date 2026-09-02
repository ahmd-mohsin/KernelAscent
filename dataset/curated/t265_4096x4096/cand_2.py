import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 265
M, D, DT = 4096, 4096, torch.bfloat16


@triton.jit
def _fused_relu_scale_softmax_kernel(
    X_ptr, Out_ptr,
    N, stride_x, stride_o,
    C1, C2,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)

    # relu (bf16), then two scalar multiplies each rounded to bf16
    # to match the reference elementwise bf16 arithmetic exactly.
    x = tl.maximum(x, 0.0)
    t = x.to(tl.float32) * C1
    t = t.to(tl.bfloat16).to(tl.float32)
    # second relu is a no-op (values are nonnegative)
    t = t * C2
    t = t.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matching PyTorch's fp32 accumulation for bf16 inputs)
    t = tl.where(mask, t, float('-inf'))
    row_max = tl.max(t, axis=0)
    e = tl.exp(t - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out_ptr + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (already optimal on A100 tensor cores)
        y = x @ self.W0

        orig_shape = y.shape
        y2 = y.reshape(-1, orig_shape[-1])
        rows, N = y2.shape
        out = torch.empty_like(y2)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16

        _fused_relu_scale_softmax_kernel[(rows,)](
            y2, out,
            N, y2.stride(0), out.stride(0),
            1.3853, 1.1094,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out.reshape(orig_shape)
