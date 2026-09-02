import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 112
M, D, DT = 4096, 2048, torch.bfloat16

_INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_gelu2_scale_softmax(
    X_ptr, Y_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu #1 (exact/erf), round back to bf16 like the reference does
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # gelu #2
    g = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale (opmath fp32, store bf16 like reference)
    g = g * 1.1137
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch's internal fp32 math)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul with TF32/bf16 tensor cores on A100
        h = x @ self.W0

        rows, n = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu2_scale_softmax[(rows,)](
            h, out,
            n, h.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
