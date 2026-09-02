import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 362
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_softmax_scale_bias_softmax(
    X_ptr, B_ptr, Out_ptr,
    N, stride_x, stride_o,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load one row (bf16 -> fp32 accumulation, matching PyTorch softmax internals)
    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # First softmax (fp32 math, result rounded to bf16 like PyTorch would)
    m1 = tl.max(x, axis=0)
    e1 = tl.exp(x - m1)
    e1 = tl.where(mask, e1, 0.0)
    s1 = tl.sum(e1, axis=0)
    p = (e1 / s1).to(tl.bfloat16)

    # x * 1.2039 (fp32 opmath, rounded back to bf16 — matches PyTorch elementwise)
    y = (p.to(tl.float32) * SCALE).to(tl.bfloat16)

    # x + b3 (fp32 opmath, rounded back to bf16)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (y.to(tl.float32) + b).to(tl.bfloat16)

    # Second softmax
    zf = tl.where(mask, z.to(tl.float32), float('-inf'))
    m2 = tl.max(zf, axis=0)
    e2 = tl.exp(zf - m2)
    e2 = tl.where(mask, e2, 0.0)
    s2 = tl.sum(e2, axis=0)
    q = (e2 / s2).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_o + offs, q, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        rows, cols = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(cols)
        _fused_softmax_scale_bias_softmax[(rows,)](
            h, self.b3, out,
            cols, h.stride(0), out.stride(0),
            SCALE=1.2039,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
