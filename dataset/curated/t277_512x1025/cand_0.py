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
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    # relu (exact in any precision), then compute softmax in fp32 (matches PyTorch half softmax)
    xf = x.to(tl.float32)
    xf = tl.maximum(xf, 0.0)
    xf = tl.where(mask, xf, float('-inf'))

    row_max = tl.max(xf, axis=0)
    e = tl.math.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to fp16 (softmax output dtype), then scale in fp32 opmath, cast back (matches eager)
    p16 = p.to(tl.float16)
    out = (p16.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 1024, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 1024 else 4
        _relu_softmax_scale_kernel[(m,)](
            h, y,
            h.stride(0), y.stride(0),
            n,
            SCALE=1.1113,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
