import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 930
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_epilogue_softmax(
    X_ptr, B_ptr, Out_ptr,
    stride_x, stride_o,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.0855  (rounded to bf16 like the reference)
    x = x * 1.0855
    x = x.to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))), rounded to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # relu
    g = tl.maximum(g, 0.0)

    # + bias, rounded to bf16
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = g + b
    y = y.to(tl.bfloat16).to(tl.float32)

    # relu
    y = tl.maximum(y, 0.0)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16 softmax)
    y_masked = tl.where(mask, y, float('-inf'))
    m = tl.max(y_masked, axis=0)
    e = tl.exp(y_masked - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out_ptr + row * stride_o + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (bf16)
        h = x @ self.W0  # (M, 2048), bf16

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_epilogue_softmax[(Mrows,)](
            h, self.b4, out,
            h.stride(0), out.stride(0),
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
