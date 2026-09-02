import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 531
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _scale_relu_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N, SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # scale computed at fp32 then rounded back to bf16 (matches PyTorch opmath)
    xs = (x.to(tl.float32) * SCALE).to(tl.bfloat16)
    # relu in bf16
    zero = tl.zeros((BLOCK_N,), dtype=tl.bfloat16)
    xr = tl.maximum(xs, zero)
    # softmax in fp32 (matches PyTorch bf16 softmax accumulate type)
    xf = tl.where(mask, xr.to(tl.float32), float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 GEMM with fp32 accumulation
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _scale_relu_softmax_kernel[(Mrows,)](
            h, y,
            h.stride(0), y.stride(0),
            N, 1.3067,
            BLOCK_N=BLOCK_N,
            num_warps=8 if BLOCK_N >= 512 else 4,
        )
        return y
