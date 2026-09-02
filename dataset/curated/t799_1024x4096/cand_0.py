import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 799
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_ln_gelu_softmax(
    X, OUT, B1, G, Bt, B4,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    base = row * N

    x = tl.load(X + base + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)

    # x + b1, rounded to fp16 like the reference
    x = (x.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)

    # LayerNorm in fp32
    xf = x.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(Bt + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd) * g + bt
    y = y.to(tl.float16)

    # exact GELU in fp32, then round to fp16
    yf = y.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    gel = yf * 0.5 * (1.0 + tl.math.erf(yf * INV_SQRT2))
    gel = gel.to(tl.float16)

    # + b4, rounded to fp16
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    z = (gel.to(tl.float32) + b4.to(tl.float32)).to(tl.float16)

    # softmax in fp32
    zf = tl.where(mask, z.to(tl.float32), float('-inf'))
    zmax = tl.max(zf, axis=0)
    e = tl.exp(zf - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + base + cols, out, mask=mask)


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
        h = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(n)
        _fused_ln_gelu_softmax[(m,)](
            h, out, self.b1, self.ln2_g, self.ln2_b, self.b4,
            n, 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out
