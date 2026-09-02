import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 926
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_bias_relu_scale_softmax(
    X, B, OUT,
    stride_xm,
    stride_om,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # x + b0 computed in fp32 (opmath), rounded to bf16 (matches eager bf16 add)
    y = (x + b).to(tl.bfloat16).to(tl.float32)
    # relu (exact in any precision)
    y = tl.maximum(y, 0.0)
    # scalar multiply in fp32, rounded to bf16 (matches eager bf16 mul)
    y = (y * SCALE).to(tl.bfloat16).to(tl.float32)

    # softmax in fp32 (matches PyTorch bf16 softmax which upcasts to float)
    y = tl.where(mask, y, float("-inf"))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(OUT + row * stride_om + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x + self.b0
            y = torch.relu(y)
            y = y * 1.0557
            return torch.softmax(y, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        out = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _fused_bias_relu_scale_softmax[(m,)](
            x2, self.b0, out,
            x2.stride(0), out.stride(0),
            n, BLOCK_N, 1.0557,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
