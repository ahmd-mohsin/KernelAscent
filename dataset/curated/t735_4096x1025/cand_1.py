import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 735
M, D, DT = 4096, 1025, torch.bfloat16


@triton.jit
def _bias_scale_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # replicate: (x + b) in fp32 -> round bf16 ; (* scale) in fp32 -> round bf16
    y = (x + b).to(tl.bfloat16).to(tl.float32)
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride_y + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (same as reference)
        h = x @ self.W0
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _bias_scale_softmax_kernel[(rows,)](
            h, self.b1, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.1961,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
