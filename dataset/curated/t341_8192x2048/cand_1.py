import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 341
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_relu_gelu_softmax(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # exact gelu (erf-based), computed in fp32 like PyTorch's opmath for half
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))

    # round-trip through fp16 to match PyTorch pipeline (gelu output stored as fp16)
    g = g.to(tl.float16).to(tl.float32)

    # softmax over the row (fp32 accumulation, matching PyTorch half softmax)
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y_ptr + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS tensor-core matmul
        h = x @ self.W0

        h = h.contiguous()
        rows, cols = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_gelu_softmax[(rows,)](
            h, out,
            cols, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
