import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 603
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _scale_bias_softmax_kernel(
    Y_ptr, B_ptr, O_ptr,
    N, stride_y, stride_o,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    y = tl.load(Y_ptr + row * stride_y + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # emulate reference fp16 arithmetic: (x * scale) rounded to fp16, then + b rounded to fp16
    v = (y.to(tl.float32) * SCALE).to(tl.float16)
    v = (v.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    # softmax in fp32 (matches PyTorch half softmax accumulation)
    x = v.to(tl.float32)
    x = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(O_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _scale_bias_softmax_kernel[(Mrows,)](
            y, self.b2, out,
            N, y.stride(0), out.stride(0),
            SCALE=1.2045,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
