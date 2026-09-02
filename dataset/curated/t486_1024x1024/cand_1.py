import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 486
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_gelu_bias_softmax(
    X, B1, B2, Out,
    stride_xm, stride_om,
    N, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)

    # GELU (exact, erf-based) computed in fp32 like PyTorch, then cast back to fp16
    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # bias adds in fp16 to match eager arithmetic
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)
    y = (g16 + b1) + b2

    # softmax in fp32 accumulation (matches PyTorch half softmax)
    yf = tl.where(mask, y.to(tl.float32), float('-inf'))
    row_max = tl.max(yf, axis=0)
    num = tl.math.exp(yf - row_max)
    denom = tl.sum(num, axis=0)
    out = num / denom

    tl.store(Out + row * stride_om + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = x + self.b1
            x = x + self.b2
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_gelu_bias_softmax[(m,)](
            x, self.b1, self.b2, out,
            x.stride(0), out.stride(0),
            n, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
