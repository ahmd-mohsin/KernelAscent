import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 113
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _relu_softmax_kernel(
    X, Y,
    N,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # fused relu
    x = tl.maximum(x, 0.0)
    # mask invalid lanes to -inf so they don't affect the softmax
    x = tl.where(mask, x, float('-inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100 for bf16)
        h = x @ self.W0
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        # relu -> softmax fused in one kernel; final relu is a no-op since
        # softmax outputs are always nonnegative.
        _relu_softmax_kernel[(rows,)](
            h, out,
            N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
