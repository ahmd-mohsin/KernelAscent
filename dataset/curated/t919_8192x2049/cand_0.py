import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 919
M, D, DT = 8192, 2049, torch.float16


@triton.jit
def _bias_scale_softmax_kernel(
    Y, B1, B2, OUT,
    N, stride_y, stride_o,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    y = tl.load(Y + row * stride_y + cols, mask=mask, other=0.0).to(tl.float32)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)

    # Replicate fp16 rounding at each elementwise step (PyTorch computes in
    # fp32 opmath and rounds to fp16 after each op).
    x = (y + b1).to(tl.float16).to(tl.float32)
    x = (x + b2).to(tl.float16).to(tl.float32)
    x = (x * SCALE).to(tl.float16).to(tl.float32)

    # Softmax in fp32 (matches PyTorch's fp32 accumulation for half inputs)
    x_masked = tl.where(mask, x, float('-inf'))
    row_max = tl.max(x_masked, axis=0)
    e = tl.exp(x_masked - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_scale_softmax_kernel[(m,)](
            y, self.b1, self.b2, out,
            n, y.stride(0), out.stride(0),
            SCALE=1.4531,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
