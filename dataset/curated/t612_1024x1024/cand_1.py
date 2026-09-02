import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 612
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_scale_gelu_softmax(
    X_ptr, Y_ptr,
    N, stride_row,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # scale
    x = x * SCALE

    # exact GELU (matches F.gelu default, computed in fp32 opmath like PyTorch)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

    # cast back to fp16 (intermediate tensor dtype), then upcast for softmax
    g = g.to(tl.float16).to(tl.float32)

    # softmax with masking
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y_ptr + row * stride_row + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 2048, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # matmul via cuBLAS tensor cores
        h = x @ self.W0  # (M, 2048), fp16, contiguous

        rows, cols = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_scale_gelu_softmax[(rows,)](
            h, out,
            cols, h.stride(0),
            SCALE=1.0116,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
