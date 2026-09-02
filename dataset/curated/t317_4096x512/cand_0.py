import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 317
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _gelu_relu_softmax_kernel(
    X_ptr, Y_ptr,
    N, stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # GELU (erf-based, computed in fp32 like PyTorch opmath), rounded to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16)

    # ReLU (exact in bf16)
    v = tl.maximum(g, 0.0).to(tl.float32)

    # Softmax with fp32 accumulation (matches PyTorch bf16 softmax)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, 0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, 0)
    out = e / s

    tl.store(Y_ptr + row * stride + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 tensor-core matmul
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _gelu_relu_softmax_kernel[(Mrows,)](
            h, y, N, h.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
