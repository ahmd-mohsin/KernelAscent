import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 123
M, D, DT = 512, 2049, torch.bfloat16


@triton.jit
def _softmax_rms_relu_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 math, output rounded to bf16 to match reference)
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom
    s_bf16 = s.to(tl.bfloat16)
    sf = s_bf16.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, sf * sf, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    y = (sf * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = y * w

    # relu (idempotent, apply once)
    y = tl.maximum(y, 0.0)

    tl.store(Out_ptr + row * stride_xm + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # (M, 512) bf16 GEMM via cuBLAS/tensor cores
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(n)
        _softmax_rms_relu_kernel[(m,)](
            h, self.rms2_w, out,
            h.stride(0),
            n, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return out
