import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 726
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _fused_scale_bias_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # x * scale in fp32, round to bf16 (matches PyTorch opmath behavior)
    x = (x.to(tl.float32) * scale).to(tl.bfloat16)
    # add bias in fp32, round to bf16
    x = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax in fp32
    xf = x.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    row_max = tl.max(xf, axis=0)
    num = tl.exp(xf - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _fused_scale_bias_softmax[(m,)](
            x, self.b1, y,
            x.stride(0), y.stride(0),
            n, 1.0428,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y
