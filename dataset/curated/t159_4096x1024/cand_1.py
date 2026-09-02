import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 159
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _relu_scale_softmax_kernel(
    X, Y,
    N, stride_xm, stride_ym,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    xf = x.to(tl.float32)
    # relu
    xf = tl.where(xf > 0.0, xf, 0.0)
    # scale (compute in fp32, round to fp16 to match PyTorch half elementwise semantics)
    y16 = (xf * SCALE).to(tl.float16)
    yf = tl.where(mask, y16.to(tl.float32), float('-inf'))

    # numerically stable softmax in fp32
    row_max = tl.max(yf, axis=0)
    e = tl.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        z = torch.matmul(x, self.W0)  # cuBLAS fp16 GEMM (tensor cores)
        m, n = z.shape
        out = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _relu_scale_softmax_kernel[(m,)](
            z, out,
            n, z.stride(0), out.stride(0),
            SCALE=1.238,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
