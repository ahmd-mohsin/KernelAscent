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
    X, B1, G, B, B4, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)

    x = tl.load(X + row * stride_x + cols).to(tl.float32)
    b1 = tl.load(B1 + cols).to(tl.float32)

    # x + b1 (rounded to fp16 as in reference)
    y = (x + b1).to(tl.float16).to(tl.float32)

    # LayerNorm (fp32 accumulation, fp16 output)
    mean = tl.sum(y, axis=0) / N
    d = y - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    z = (d * rstd * g + b).to(tl.float16).to(tl.float32)

    # exact GELU (erf), fp32 math, fp16 output
    ge = (0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))).to(tl.float16).to(tl.float32)

    # + b4 (fp16 rounding)
    b4 = tl.load(B4 + cols).to(tl.float32)
    w = (ge + b4).to(tl.float16).to(tl.float32)

    # softmax (fp32 accumulation, fp16 output)
    m = tl.max(w, axis=0)
    e = tl.exp(w - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out)


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
        h = x @ self.W0  # cuBLAS fp16 GEMM with fp32 accumulation
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_ln_gelu_softmax[(m,)](
            h, self.b1, self.ln2_g, self.ln2_b, self.b4, out,
            h.stride(0), out.stride(0),
            N=n, BLOCK=n,
            num_warps=4,
        )
        return out
