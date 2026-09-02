import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 676
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_epilogue_softmax(
    X_ptr, B_ptr, Y_ptr,
    N, stride_row,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    # load matmul result row (bf16), compute in fp32 like PyTorch opmath
    x = tl.load(X_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)

    # relu
    x = tl.maximum(x, 0.0)

    # exact (erf) gelu in fp32, then round to bf16 (matches PyTorch's gelu on bf16)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # bias add in fp32, round to bf16 (matches bf16 elementwise add)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = (g + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax which accumulates in fp32)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_row + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 GEMM (identical to reference matmul)
        y = x @ self.W0

        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _fused_epilogue_softmax[(Mrows,)](
            y, self.b3, out,
            N, y.stride(0),
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
