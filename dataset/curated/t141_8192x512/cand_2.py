import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 141
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_relu_bias_relu_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # relu -> add bias (bias added in bf16 precision domain: emulate x.relu() + b in bf16)
    x = tl.maximum(x, 0.0)
    v = x + b
    # round to bf16 to match reference numerics (add happens in bf16)
    v = v.to(tl.bfloat16).to(tl.float32)
    v = tl.maximum(v, 0.0)

    # softmax in fp32 (matches PyTorch internal upcast)
    v = tl.where(mask, v, float('-inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_relu_bias_relu_softmax[(Mrows,)](
            x, self.b1, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y
