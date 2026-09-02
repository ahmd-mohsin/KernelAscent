import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 342
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _double_softmax_kernel(X, Y, N, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # first softmax (fp32 accumulate, like PyTorch's half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    # round to fp16 to match the intermediate tensor of the reference
    y = y.to(tl.float16).to(tl.float32)

    # second softmax
    m2 = tl.max(tl.where(mask, y, float('-inf')), axis=0)
    e2 = tl.exp(y - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    out = e2 / s2

    tl.store(Y + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores on A100)
        h = h.contiguous()
        rows, cols = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(cols)
        _double_softmax_kernel[(rows,)](
            h, out, cols, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=4,
        )
        return out
