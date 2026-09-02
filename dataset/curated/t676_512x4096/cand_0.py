import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 676
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_relu_gelu_bias_softmax(
    X_ptr, B_ptr, Out_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load matmul output row (bf16 -> fp32, exact)
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # ReLU (exact in any precision)
    x = tl.maximum(x, 0.0)

    # GELU (erf-based, computed in fp32 like PyTorch's opmath, rounded to bf16)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # Bias add: bf16 + bf16 -> compute fp32, round to bf16 (matches PyTorch)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (g + b).to(tl.bfloat16).to(tl.float32)

    # Softmax with fp32 accumulation (matches PyTorch's acc_type behavior)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Out_ptr + row * N + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 2048, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0  # (M, 2048) bf16
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_relu_gelu_bias_softmax[(Mrows,)](
            h, self.b3, out,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
