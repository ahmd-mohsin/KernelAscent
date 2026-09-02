import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 336
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_gelu_bias_softmax(
    X, B1, B2, Y,
    N, stride_xm, stride_ym,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # exact GELU in fp32, round back to bf16 (matches F.gelu on bf16)
    xf = x.to(tl.float32)
    g = (0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))).to(tl.bfloat16)

    # bias adds in bf16 (matches reference intermediate rounding)
    h = (g + b1) + b2

    # softmax with fp32 accumulation
    hf = tl.where(mask, h.to(tl.float32), float('-inf'))
    m = tl.max(hf, axis=0)
    e = tl.exp(hf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x + self.b1
            x = x + self.b2
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu_bias_softmax[(m,)](
            x, self.b1, self.b2, y,
            n, x.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y
