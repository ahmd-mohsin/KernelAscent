import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 936
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _gelu_relu_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf) GELU in fp32, then round to fp16 (matches PyTorch half opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # relu in fp16
    r16 = tl.maximum(g16, tl.zeros_like(g16))

    # RMS norm computed in fp32 on the fp16-rounded values
    rf = r16.to(tl.float32)
    ms = tl.sum(rf * rf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    y16 = (rf * inv).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y16.to(tl.float32) * w).to(tl.float16)

    tl.store(Out_ptr + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS/tensor-core GEMM, fp16
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _gelu_relu_rms_kernel[(Mrows,)](
            h, self.rms3_w, out,
            N, h.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
