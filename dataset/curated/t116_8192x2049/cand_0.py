import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 116
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _fused_relu_double_softmax(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    # ReLU
    x = tl.maximum(x, 0.0)
    # mask out-of-range for softmax
    x = tl.where(mask, x, float('-inf'))

    # First softmax (float32 accumulation, like PyTorch's half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y1 = e1 / s1

    # Round to fp16 (intermediate tensor dtype in reference), back to fp32
    y1 = y1.to(tl.float16).to(tl.float32)
    y1 = tl.where(mask, y1, float('-inf'))

    # Second softmax
    m2 = tl.max(y1, axis=0)
    e2 = tl.exp(y1 - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    y2 = e2 / s2

    tl.store(Y_ptr + row * stride_y + offs, y2.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 GEMM (tensor cores on A100)
        h = x @ self.W0
        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_double_softmax[(Mrows,)](
            h, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
