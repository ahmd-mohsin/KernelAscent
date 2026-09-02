import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 480
M, D, DT = 4096, 2048, torch.bfloat16


@triton.jit
def _fused_relu_scale_bias_softmax(
    X, B, Y,
    stride_xm, stride_ym,
    N, SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf'))
    b = tl.load(B + cols, mask=mask, other=0.0)

    # relu (bf16), then scale in fp32 rounded back to bf16 (matches PyTorch bf16 mul)
    xf = tl.maximum(x.to(tl.float32), 0.0)
    xf = (xf * SCALE).to(tl.bfloat16).to(tl.float32)
    # add bias, round to bf16 (matches PyTorch bf16 add)
    z = (xf + b.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    z = tl.where(mask, z, float('-inf'))

    # softmax in fp32 accumulation
    zmax = tl.max(z, axis=0)
    e = tl.exp(z - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_relu_scale_bias_softmax[(m,)](
            x, self.b2, y,
            x.stride(0), y.stride(0),
            n, 1.4204,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
