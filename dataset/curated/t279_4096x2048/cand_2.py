import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 279
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_epilogue_softmax(
    X, B, OUT,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)

    # relu -> scale -> bias
    x = tl.maximum(x, 0.0) * 1.2475 + b
    # exact GELU (erf-based, matches F.gelu default)
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x * 1.2604

    # softmax over the row (masked lanes -> -inf)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    y = e / s

    tl.store(OUT + row * stride_o + offs, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Tensor-core GEMM via cuBLAS
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue_softmax[(m,)](
            h, self.b3, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
