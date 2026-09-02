import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 606
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_relu_bias_relu_softmax(
    Y, B, OUT,
    N,
    stride_ym, stride_om,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    y = tl.load(Y + row * stride_ym + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # relu -> +bias -> relu   (in bf16, matching the reference elementwise ops)
    z = tl.maximum(y, 0.0)
    z = z + b
    z = tl.maximum(z, 0.0)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    row_max = tl.max(zf, axis=0)
    num = tl.exp(zf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(OUT + row * stride_om + cols, out.to(OUT.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS bf16 GEMM
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_relu_bias_relu_softmax[(Mrows,)](
            y, self.b2, out,
            N,
            y.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
