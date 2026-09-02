import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 968
M, D, DT = 512, 2048, torch.float16


@triton.jit
def fused_relu_scale_bias_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # relu -> scale -> bias (compute in fp16-equivalent path but fp32 math;
    # replicate fp16 rounding at each step for numerical equivalence)
    x = tl.maximum(x, 0.0)
    x = (x * scale).to(tl.float16).to(tl.float32)
    x = (x + b).to(tl.float16).to(tl.float32)

    x = tl.where(mask, x, float('-inf'))
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    y = e / s

    tl.store(Y + row * stride_ym + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.0064
            x = x + self.b2
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        fused_relu_scale_bias_softmax[(m,)](
            x, self.b2, y,
            x.stride(0), y.stride(0),
            n, 1.0064,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
