import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 277
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _relu_softmax_scale_kernel(
    X, Y,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=float('-inf'))
    # relu in fp16 (matches reference), then softmax math in fp32
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))

    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)

    # softmax output cast to fp16 (as torch.softmax on fp16 tensor does),
    # then scalar multiply computed in fp32 and cast back to fp16.
    sm16 = (e / denom).to(tl.float16)
    out = (sm16.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        # tensor-core GEMM (same as reference)
        h = x @ self.W0
        h = h.contiguous()

        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _relu_softmax_scale_kernel[(m,)](
            h, y,
            n, h.stride(0), y.stride(0),
            SCALE=1.1113,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
