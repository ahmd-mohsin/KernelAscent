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
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # bf16
    b = tl.load(B + cols, mask=mask, other=0.0)                    # bf16

    # relu in bf16
    x = tl.maximum(x, 0.0)
    # mul with fp32 opmath, round back to bf16 (matches PyTorch semantics)
    x = (x.to(tl.float32) * 1.4204).to(tl.bfloat16)
    # add with fp32 opmath, round back to bf16
    x = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    # softmax computed in fp32 (matches PyTorch's internal upcast)
    xf = tl.where(mask, x.to(tl.float32), float('-inf'))
    m = tl.max(xf, axis=0)
    e = tl.exp(xf - m)
    s = tl.sum(e, axis=0)
    y = (e / s).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


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
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_relu_scale_bias_softmax[(m,)](
            x, self.b2, y,
            x.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
