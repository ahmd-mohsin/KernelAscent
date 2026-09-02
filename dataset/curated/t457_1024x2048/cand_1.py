import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 457
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _scale_rms_gelu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    stride_x, stride_y,
    eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # load bf16 input row
    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)

    # x = x * 1.2772  (fp32 opmath, rounded back to bf16, matching PyTorch)
    xf = x.to(tl.float32) * scale
    xb = xf.to(tl.bfloat16)
    xf = xb.to(tl.float32)

    # RMS statistics in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + eps)

    # normalize in fp32, round to bf16 (matches (_xf * rsqrt).to(bf16))
    n = (xf * r).to(tl.bfloat16).to(tl.float32)

    # multiply by rms weight (fp32 opmath then round -> identical to bf16 elementwise mul)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (n * w).to(tl.bfloat16).to(tl.float32)

    # exact (erf-based) GELU in fp32 opmath, as PyTorch does for bf16 on CUDA
    g = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y_ptr + row * stride_y + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = x @ self.W0  # (M, 1024) bf16

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _scale_rms_gelu_kernel[(Mrows,)](
            h, self.rms2_w, out,
            N,
            h.stride(0), out.stride(0),
            1e-6, 1.2772,
            BLOCK=BLOCK,
            num_warps=8,
        )

        # GEMM 2 (cuBLAS tensor cores)
        return out @ self.W4
