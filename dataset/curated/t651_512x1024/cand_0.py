import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 651
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _rms_relu_softmax_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.float16)  # cast to fp16 (matches .to(x.dtype))

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # fp16
    y = xn * w  # fp16 multiply (matches reference rounding)

    # ReLU in fp16
    y = tl.maximum(y, y * 0)

    # Softmax with fp32 accumulation (matches PyTorch half softmax)
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        _rms_relu_softmax_kernel[(m,)](
            x, self.rms1_w, out,
            x.stride(0),
            n, BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
