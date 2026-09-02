import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 954
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _fused_ln_softmax_kernel(
    X, G, B, Out,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
    SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # scale in fp16 (match reference: x = x * 1.0315 in fp16)
    x = (x * SCALE).to(tl.float16)
    xf = x.to(tl.float32)

    # layernorm in fp32
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b

    # cast to fp16 (LN output dtype), then softmax with fp32 accumulation
    y = y.to(tl.float16).to(tl.float32)
    y = tl.where(mask, y, float('-inf'))
    m = tl.max(y, axis=0)
    e = tl.exp(y - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s
    # relu is a no-op on softmax outputs (all >= 0)

    tl.store(Out + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_ln_softmax_kernel[(m,)](
            y, self.ln2_g, self.ln2_b, out,
            y.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK, EPS=1e-5, SCALE=1.0315,
            num_warps=4,
        )
        return out
