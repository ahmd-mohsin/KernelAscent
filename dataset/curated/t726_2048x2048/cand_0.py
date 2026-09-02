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
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # x * scale : computed in fp32, rounded to bf16 (matches PyTorch opmath)
    y = (x.to(tl.float32) * scale).to(tl.bfloat16)
    # y + b : computed in fp32, rounded to bf16
    z = (y.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax with fp32 accumulation (matches PyTorch bf16 softmax)
    zf = tl.where(mask, z.to(tl.float32), float("-inf"))
    m = tl.max(zf, axis=0)
    e = tl.exp(zf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_scale_bias_softmax[(Mrows,)](
            x, self.b1, y,
            x.stride(0), y.stride(0),
            N, 1.0428,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
