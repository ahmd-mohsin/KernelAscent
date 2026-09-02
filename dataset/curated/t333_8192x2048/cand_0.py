import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 333
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _scale_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xm,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)  # bf16
    # x * 1.4797 : computed in fp32 then cast back to bf16 (matches PyTorch scalar mul)
    xs = (x.to(tl.float32) * SCALE).to(tl.bfloat16)

    xf = xs.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    xn = (xf * inv).to(tl.bfloat16)  # cast to bf16 first
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)  # bf16
    y = xn * w  # bf16 * bf16 multiply (Triton promotes to fp32 internally, result cast)
    tl.store(Y_ptr + row * stride_xm + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(n)
        _scale_rmsnorm_kernel[(m,)](
            y, self.rms2_w, out,
            y.stride(0),
            n, 1.4797, 1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return out
