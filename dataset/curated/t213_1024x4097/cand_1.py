import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 213
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _rms_gelu_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, eps, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS in fp32 (matches torch: float() -> pow(2).mean -> rsqrt)
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + eps)

    # normalize in fp32, round to fp16 (matches .to(x.dtype))
    xn = (x * r).to(tl.float16)

    # multiply by weight: torch half elementwise mul computes in fp32 (opmath), rounds to fp16
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn.to(tl.float32) * w).to(tl.float16)

    # scalar multiply: computed in fp32, rounded to fp16
    z = (y.to(tl.float32) * scale).to(tl.float16)

    # exact (erf-based) GELU, computed in fp32, rounded to fp16
    g = z.to(tl.float32)
    out = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))

    tl.store(Y_ptr + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores (same as reference)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _rms_gelu_kernel[(Mrows,)](
            h, self.rms1_w, out,
            N, 1e-6, 1.2985,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
