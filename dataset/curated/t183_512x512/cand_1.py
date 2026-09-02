import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 183
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_bias_gelu_softmax(
    X, B1, B2, B3, OUT,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)
    b3 = tl.load(B3 + offs, mask=mask, other=0.0)

    # sequential fp16 rounding to match reference add order
    x = (x + b1).to(tl.float16)
    x = (x + b2).to(tl.float16)
    x = (x + b3).to(tl.float16)

    xf = x.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = g.to(tl.float16).to(tl.float32)

    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x @ self.W0
            y = y + self.b1
            y = y + self.b2
            y = y + self.b3
            y = F.gelu(y)
            return torch.softmax(y, dim=-1)

        # cuBLAS fp16 GEMM (tensor cores)
        y = torch.matmul(x, self.W0)
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_bias_gelu_softmax[(Mrows,)](
            y, self.b1, self.b2, self.b3, out,
            N, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
