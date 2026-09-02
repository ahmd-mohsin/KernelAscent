import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 395
M, D, DT = 2048, 2049, torch.float16


@triton.jit
def _gelu_softmax_kernel(
    X, Y,
    N, stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)

    # GELU (erf variant) computed in fp32 (opmath), then cast back to fp16
    xf = x.to(tl.float32)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # Softmax with fp32 accumulation (matches PyTorch half softmax)
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, float('-inf'))
    row_max = tl.max(gf, axis=0)
    num = tl.exp(gf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)

    def forward(self, x):
        h = torch.matmul(x, self.W0)  # cuBLAS TF32/FP16 tensor-core GEMM
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _gelu_softmax_kernel[(Mrows,)](
            h, y, N, h.stride(0), y.stride(0),
            BLOCK_N=BLOCK_N, num_warps=num_warps,
        )
        return y
