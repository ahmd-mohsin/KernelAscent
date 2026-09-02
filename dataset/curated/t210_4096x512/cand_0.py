import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 210
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _gelu_softmax_kernel(
    X_ptr, Y_ptr,
    stride_x, stride_y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact (erf-based) GELU, computed in fp32 then rounded to bf16
    # to match PyTorch's bf16 gelu kernel behavior
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # softmax with fp32 accumulation (matches PyTorch softmax on bf16)
    g = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    y = e / denom

    tl.store(Y_ptr + row * stride_y + cols, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W1 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # scale in bf16 first (matches reference numerics), then cuBLAS matmul
        h = (x * 1.02) @ self.W1

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 1024 else 8
        _gelu_softmax_kernel[(Mrows,)](
            h, out,
            h.stride(0), out.stride(0),
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
