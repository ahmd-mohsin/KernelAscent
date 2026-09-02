import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 309
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _double_softmax_bias_kernel(
    X, B, Y,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # first softmax (fp32 accumulate, matching PyTorch half softmax)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    s1 = tl.sum(e1, axis=0)
    y1 = e1 / s1
    # round intermediate to fp16 as PyTorch produces a fp16 tensor between softmaxes
    y1 = y1.to(tl.float16).to(tl.float32)
    y1 = tl.where(mask, y1, float('-inf'))

    # second softmax
    m2 = tl.max(y1, axis=0)
    e2 = tl.exp(y1 - m2)
    s2 = tl.sum(e2, axis=0)
    y2 = (e2 / s2).to(tl.float16)

    b = tl.load(B + cols, mask=mask, other=0.0)
    out = y2 + b

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 4096, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _double_softmax_bias_kernel[(Mrows,)](
            h, self.b3, out,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
