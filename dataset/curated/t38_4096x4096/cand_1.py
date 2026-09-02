import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 38
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _double_softmax_kernel(
    X_ptr, Out_ptr,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # First softmax (fp32 accumulate, like PyTorch on fp16 input)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    y = e1 / s1

    # Reference stores intermediate as fp16 -> round to fp16 then back to fp32
    y = y.to(tl.float16).to(tl.float32)
    y = tl.where(mask, y, float('-inf'))

    # Second softmax
    m2 = tl.max(y, axis=0)
    e2 = tl.exp(y - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    z = e2 / s2

    tl.store(Out_ptr + row * stride_o + cols, z.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores on A100)
        y = x @ self.W0
        y = y.contiguous()
        rows, n = y.shape
        out = torch.empty_like(y)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _double_softmax_kernel[(rows,)](
            y, out,
            n, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
