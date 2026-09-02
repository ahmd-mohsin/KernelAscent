import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 8
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _gelu_softmax_bias_kernel(
    X, B, Y,
    stride_xm, stride_ym,
    N, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact GELU in fp32 (matches PyTorch opmath for half), round to fp16
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)

    # softmax with fp32 accumulation (matches PyTorch half softmax)
    gf = tl.where(mask, gf, float('-inf'))
    row_max = tl.max(gf, axis=0)
    e = tl.exp(gf - row_max)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # bias add in fp16
    b = tl.load(B + cols, mask=mask, other=0.0)
    out = sm + b

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            return x + self.b2

        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _gelu_softmax_bias_kernel[(m,)](
            x, self.b2, y,
            x.stride(0), y.stride(0),
            n, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y
