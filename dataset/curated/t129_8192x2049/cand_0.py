import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 129
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _fused_relu_softmax_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load matmul output row (bf16), relu, upcast to fp32 (softmax computed in fp32)
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x = tl.maximum(x, 0.0)
    x = tl.where(mask, x, float('-inf'))

    # Softmax in fp32
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    # Cast down to bf16 (matches torch.softmax output dtype), then back up for RMS in fp32
    y = y.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(y * y, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)

    # Round normalized value to bf16, then multiply by weight (fp32 math, bf16 result)
    a = (y * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = (a * w).to(tl.bfloat16)

    tl.store(Out_ptr + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 1024, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Heavy GEMM via cuBLAS tensor cores
        h = x @ self.W0  # (M, 1024) bf16
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_relu_softmax_rms_kernel[(Mrows,)](
            h, self.rms3_w, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
