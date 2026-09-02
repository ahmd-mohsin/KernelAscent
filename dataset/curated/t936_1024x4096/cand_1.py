import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 936
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_gelu_relu_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact (erf-based) GELU in fp32, then cast to fp16 (matches PyTorch half gelu)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g16 = g.to(tl.float16)

    # relu in fp16
    r16 = tl.maximum(g16, tl.zeros_like(g16))

    # RMSNorm in fp32
    rf = r16.to(tl.float32)
    ms = tl.sum(rf * rf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    normed16 = (rf * inv).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    out = (normed16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_gelu_relu_rms_kernel[(m,)](
            y, self.rms3_w, out,
            n, y.stride(0), out.stride(0),
            1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
