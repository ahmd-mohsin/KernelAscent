import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 799
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(X, B1, G, B, B4, OUT,
                  stride_x, stride_o,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)

    # x + b1 (fp16 rounding to match reference)
    x = (x + b1).to(tl.float16)

    # layernorm in fp32
    xf = x.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    d = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd) * g + b
    y = y.to(tl.float16)

    # gelu (exact, fp32 math)
    yf = y.to(tl.float32)
    yg = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    yg = yg.to(tl.float16)

    # + b4
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    z = (yg + b4).to(tl.float16)

    # softmax in fp32
    zf = tl.where(mask, z.to(tl.float32), float('-inf'))
    zmax = tl.max(zf, axis=0)
    e = tl.exp(zf - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            y, self.b1, self.ln2_g, self.ln2_b, self.b4, out,
            y.stride(0), out.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
