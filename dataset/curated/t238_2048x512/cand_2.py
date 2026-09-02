import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 238
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _bias_softmax_kernel(
    X, B, OUT,
    stride_xm, stride_om,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bias add in fp16 to match reference (x + b1 on fp16 tensors)
    xb = (x + b).to(tl.float16)

    # softmax with fp32 accumulation (matches PyTorch half softmax acctype)
    xf = xb.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    e = tl.exp(xf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(OUT + row * stride_om + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        z = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        m, n = z.shape
        out = torch.empty_like(z)
        BLOCK_N = triton.next_power_of_2(n)
        _bias_softmax_kernel[(m,)](
            z, self.b1, out,
            z.stride(0), out.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
