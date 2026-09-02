import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 564
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _relu_bias_softmax_kernel(
    Y_ptr, B_ptr, Out_ptr,
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # relu in fp16 (exact), then fp16 add to match reference rounding
    y = tl.maximum(y, 0.0).to(tl.float16)
    y = y + b  # fp16 add, rounded like the reference

    # softmax with fp32 accumulation (matches PyTorch's fp16 softmax)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    row_max = tl.max(yf, axis=0)
    e = tl.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(Out_ptr + row * stride_row + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        # GEMM 1 (tensor cores) + in-place scale (single cheap elementwise kernel)
        h = torch.mm(x, self.W0)
        h.mul_(1.4351)

        # GEMM 2 (tensor cores)
        y = torch.mm(h, self.W2)

        # Fused relu + bias + softmax in one Triton kernel
        out = torch.empty_like(y)
        Mrows, N = y.shape
        BLOCK = triton.next_power_of_2(N)
        _relu_bias_softmax_kernel[(Mrows,)](
            y, self.b4, out,
            N, y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
