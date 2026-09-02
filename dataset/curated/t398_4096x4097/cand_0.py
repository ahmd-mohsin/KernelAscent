import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 398
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_bias_rms2_relu_kernel(
    X_ptr, B_ptr, W2_ptr, W3_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    # load matmul output (bf16) and bias (bf16)
    x = tl.load(X_ptr + row * stride_x + cols)
    b = tl.load(B_ptr + cols)

    # bias add in bf16 (matches torch bf16 elementwise semantics)
    x = x + b

    # first RMSNorm
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    w2 = tl.load(W2_ptr + cols)
    x = (xf * r).to(tl.bfloat16) * w2

    # second RMSNorm
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    w3 = tl.load(W3_ptr + cols)
    x = (xf * r).to(tl.bfloat16) * w3

    # relu
    zero = tl.zeros_like(x)
    x = tl.where(x > zero, x, zero)

    tl.store(Y_ptr + row * stride_y + cols, x)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # heavy GEMM via cuBLAS tensor cores
        h = torch.matmul(x, self.W0)
        h = h.contiguous()

        m, n = h.shape
        y = torch.empty_like(h)

        grid = (m,)
        _fused_bias_rms2_relu_kernel[grid](
            h, self.b1, self.rms2_w, self.rms3_w, y,
            h.stride(0), y.stride(0),
            N=n,
            BLOCK=triton.next_power_of_2(n),
            num_warps=8,
        )
        return y
