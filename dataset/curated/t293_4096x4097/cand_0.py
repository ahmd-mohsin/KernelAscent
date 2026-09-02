import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 293
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _gelu_bias_softmax_kernel(
    X_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU in fp32, then round to bf16 to match PyTorch elementwise op
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (g + b).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax which accumulates in fp32)
    z = tl.where(mask, z, float('-inf'))
    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y_ptr + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 1024, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _gelu_bias_softmax_kernel[(m,)](
            h, self.b2, out,
            n, h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
