import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 175
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _relu_double_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    ptr = X_ptr + row * stride + offs

    x = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    # ReLU, masked-out lanes set to -inf so they contribute 0 to softmax
    x = tl.where(mask, tl.maximum(x, 0.0), float('-inf'))

    # First softmax (fp32 accumulation, matching PyTorch half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.math.exp(x - m1)
    s1 = tl.sum(e1, axis=0)
    y = e1 / s1

    # Round-trip through fp16 to match the reference intermediate dtype
    y = y.to(tl.float16).to(tl.float32)
    y = tl.where(mask, y, float('-inf'))

    # Second softmax
    m2 = tl.max(y, axis=0)
    e2 = tl.math.exp(y - m2)
    s2 = tl.sum(e2, axis=0)
    out = e2 / s2

    tl.store(Y_ptr + row * stride + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        rows, cols = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(cols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _relu_double_softmax_kernel[(rows,)](
            h, out,
            cols, h.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
