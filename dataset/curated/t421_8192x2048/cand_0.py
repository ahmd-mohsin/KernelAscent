import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 421
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_epilogue(
    X_ptr, B_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    # load row (bf16 -> fp32)
    x = tl.load(X_ptr + row * stride_x + offs).to(tl.float32)

    # x = x * 1.1885  (bf16 rounding after op, matching bf16 elementwise semantics)
    x = (x * 1.1885).to(tl.bfloat16).to(tl.float32)

    # x = x + b2
    b = tl.load(B_ptr + offs).to(tl.float32)
    x = (x + b).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based), rounded back to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32 over the row
    ms = tl.sum(g * g, axis=0) / N
    rinv = 1.0 / tl.sqrt(ms + 1e-6)

    # (normed).to(bf16) * rms4_w  (bf16 mul == fp32 compute + bf16 round)
    y = (g * rinv).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_ptr + offs).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # final GELU
    out = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    tl.store(Y_ptr + row * stride_y + offs, out.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core matmul (bf16)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        y = torch.empty_like(h)

        _fused_epilogue[(m,)](
            h, self.b2, self.rms4_w, y,
            n, h.stride(0), y.stride(0),
            BLOCK=4096,
            num_warps=8,
            num_stages=2,
        )
        return y
