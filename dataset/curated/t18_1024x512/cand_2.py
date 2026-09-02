import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 18
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _scale_relu_scale_softmax_kernel(
    X, Y,
    N, stride_x, stride_y,
    S1, S2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.2399  (round to bf16 like the reference)
    x = (x * S1).to(tl.bfloat16).to(tl.float32)
    # relu
    x = tl.maximum(x, 0.0)
    # x = x * 1.1622  (round to bf16 like the reference)
    x = (x * S2).to(tl.bfloat16).to(tl.float32)

    # softmax (fp32 accumulation, like PyTorch's bf16 softmax)
    x = tl.where(mask, x, float('-inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS matmul (tensor cores on A100 for bf16)
        h = x @ self.W0
        h = h.contiguous()

        Mr, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _scale_relu_scale_softmax_kernel[(Mr,)](
            h, out,
            N, h.stride(0), out.stride(0),
            1.2399, 1.1622,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
