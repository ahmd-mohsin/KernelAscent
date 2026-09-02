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
    X, B1, B2, B3, Out,
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

    # sequential fp16 adds to match reference order/precision
    x = x + b1
    x = x + b2
    x = x + b3

    # GELU (exact, erf) computed in fp32 (opmath), rounded back to fp16
    xf = x.to(tl.float32)
    gf = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    g = gf.to(tl.float16).to(tl.float32)

    # softmax with fp32 accumulation
    g_masked = tl.where(mask, g, float('-inf'))
    row_max = tl.max(g_masked, axis=0)
    e = tl.exp(g - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.float16)

    tl.store(Out + row * stride_o + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # tensor-core matmul via cuBLAS
        y = torch.matmul(x, self.W0)
        rows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_bias_gelu_softmax[(rows,)](
            y, self.b1, self.b2, self.b3, out,
            N, y.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
