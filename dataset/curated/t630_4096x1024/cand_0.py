import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 630
M, D, DT = 4096, 1024, torch.float16

_INV_SQRT2 = 0.7071067811865476


@triton.jit
def _gelu_softmax_gelu_gelu_kernel(
    X_ptr, Y_ptr,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based), computed in fp32 then cast to fp16 (matches PyTorch half opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    # Softmax in fp32 (matches PyTorch half softmax which accumulates in float)
    g_masked = tl.where(mask, g, float('-inf'))
    m = tl.max(g_masked, axis=0)
    e = tl.exp(g_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.float16).to(tl.float32)

    # GELU
    p = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))
    p = p.to(tl.float16).to(tl.float32)

    # GELU
    p = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))

    tl.store(Y_ptr + row * stride_y + offs, p.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS GEMM
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()

        rows, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _gelu_softmax_gelu_gelu_kernel[(rows,)](
            h, out,
            h.stride(0), out.stride(0),
            N=n,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
