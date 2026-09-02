import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 289
M, D, DT = 2048, 1025, torch.float16


@triton.jit
def _gelu_bias_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_xm + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU in fp32, then round to fp16 like PyTorch elementwise op
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g16 = g.to(tl.float16)

    b16 = tl.load(B_ptr + offs, mask=mask, other=0.0)
    v16 = g16 + b16  # fp16 add, matching x + self.b2

    # softmax with fp32 accumulation (matches PyTorch half softmax semantics)
    v = v16.to(tl.float32)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.float16)

    tl.store(Y_ptr + row * stride_ym + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 4096, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 matmul with fp32 accumulation on A100 tensor cores
        h = torch.mm(x, self.W0)

        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _gelu_bias_softmax_kernel[(Mrows,)](
            h, self.b2, y,
            N, h.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
