import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 266
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_epilogue_ln(
    X, OUT, B1, B2, G, B,
    stride_x, stride_o,
    N: tl.constexpr,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # replicate fp16 rounding of each elementwise step
    x = (x + b1).to(tl.float16)
    x = (x + b2).to(tl.float16)
    x = tl.maximum(x, 0.0).to(tl.float16)
    x = (x * SCALE).to(tl.float16)

    # layernorm in fp32 (matches PyTorch internal upcast)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b

    tl.store(OUT + row * stride_o + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = torch.matmul(x, self.W0)
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_epilogue_ln[(m,)](
            x, out, self.b1, self.b2, self.ln5_g, self.ln5_b,
            x.stride(0), out.stride(0),
            N=n, SCALE=1.0466, EPS=1e-5, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
