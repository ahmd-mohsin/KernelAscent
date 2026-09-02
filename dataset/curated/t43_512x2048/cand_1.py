import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 43
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_epilogue_kernel(
    X_ptr, Y_ptr,
    N,
    stride_xm, stride_ym,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # GELU (exact, erf-based)
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    # Softmax 1
    x_masked = tl.where(mask, x, float('-inf'))
    m = tl.max(x_masked, axis=0)
    e = tl.exp(x_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = e / s

    # ReLU (softmax output is non-negative; exact identity but keep for safety)
    x = tl.maximum(x, 0.0)

    # Softmax 2
    x_masked = tl.where(mask, x, float('-inf'))
    m = tl.max(x_masked, axis=0)
    e = tl.exp(x_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = e / s

    # GELU
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    tl.store(Y_ptr + row * stride_ym + cols, x.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM (tensor cores)
        h = x @ self.W0
        if not h.is_cuda:
            h = F.gelu(h)
            h = torch.softmax(h, dim=-1)
            h = torch.relu(h)
            h = torch.softmax(h, dim=-1)
            return F.gelu(h)

        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_epilogue_kernel[(Mrows,)](
            h, out, N,
            h.stride(0), out.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=8 if BLOCK_N >= 512 else 4,
        )
        return out
