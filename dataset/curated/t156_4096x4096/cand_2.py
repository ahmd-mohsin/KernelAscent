import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 156
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _epilogue_softmax_kernel(
    X_ptr, B3_ptr, B4_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0)  # fp16
    b3 = tl.load(B3_ptr + offs, mask=mask, other=0.0)                 # fp16
    b4 = tl.load(B4_ptr + offs, mask=mask, other=0.0)                 # fp16

    # x * 1.3741 : PyTorch computes half ops with float opmath, then rounds to half
    t = (x.to(tl.float32) * SCALE).to(tl.float16)
    # relu
    zero = tl.zeros_like(t)
    t = tl.maximum(t, zero)
    # + b3 (float add of two half values is exact; round to half == half add)
    t = (t.to(tl.float32) + b3.to(tl.float32)).to(tl.float16)
    # + b4
    t = (t.to(tl.float32) + b4.to(tl.float32)).to(tl.float16)

    # softmax in fp32 (matches PyTorch's accscalar_t = float for half inputs)
    tf = t.to(tl.float32)
    tf = tl.where(mask, tf, float('-inf'))
    row_max = tl.max(tf, axis=0)
    e = tl.exp(tf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0  # (M, 1024) fp16, freshly allocated -> safe to overwrite

        Mrows, N = h.shape
        BLOCK = triton.next_power_of_2(N)
        grid = (Mrows,)
        _epilogue_softmax_kernel[grid](
            h, self.b3, self.b4, h,
            N, h.stride(0), h.stride(0),
            1.3741,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return h
