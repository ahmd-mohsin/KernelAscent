import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 928
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_epilogue_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    # scale in fp16 (matches half-precision elementwise mul)
    x = (x.to(tl.float32) * SCALE).to(tl.float16)
    # relu(relu(x)) == relu(x)
    x = tl.maximum(x, 0.0)
    # exact GELU computed in fp32 (opmath), cast back to fp16
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    x = g.to(tl.float16)
    # bias add in fp16
    x = x + b

    # softmax with fp32 accumulation
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(Out_ptr + row * stride_x + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 tensor-core GEMM
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_epilogue_softmax[(Mrows,)](
            y, self.b5, out,
            N, y.stride(0),
            SCALE=1.0502,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
