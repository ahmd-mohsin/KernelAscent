import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 421
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_epilogue_kernel(
    X_ptr, B2_ptr, W_ptr, Y_ptr,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.1885  (bf16 op: fp32 compute, round to bf16)
    x = (x * 1.1885).to(tl.bfloat16).to(tl.float32)

    # x = x + b2
    b = tl.load(B2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # x = gelu(x)  (exact, erf-based)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # rmsnorm in fp32
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x = (x * r).to(tl.bfloat16).to(tl.float32)

    # x = x * rms4_w
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w).to(tl.bfloat16).to(tl.float32)

    # x = gelu(x)
    x = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))

    tl.store(Y_ptr + row * N + offs, x.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core matmul
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue_kernel[(m,)](
            h, self.b2, self.rms4_w, y,
            n,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
