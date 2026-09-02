import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 613
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_post_kernel(
    X, B1, RW, B4, LG, LB, OUT,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    # x + b1 (bf16 elementwise add -> computed in fp32, rounded to bf16)
    x = tl.load(X + base + offs).to(tl.float32)
    b1 = tl.load(B1 + offs).to(tl.float32)
    t = (x + b1).to(tl.bfloat16)
    tf = t.to(tl.float32)

    # RMSNorm in fp32, cast to bf16, then * rms2_w (bf16 result)
    ms = tl.sum(tf * tf, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    v = (tf * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(RW + offs).to(tl.float32)
    u = (v * w).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), fp32 compute, round to bf16
    g = 0.5 * u * (1.0 + tl.math.erf(u * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # + b4 (bf16 result)
    b4 = tl.load(B4 + offs).to(tl.float32)
    h = (g + b4).to(tl.bfloat16).to(tl.float32)

    # LayerNorm in fp32 (eps = 1e-5), affine, cast to bf16
    mean = tl.sum(h, axis=0) / N
    d = h - mean
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)
    lg = tl.load(LG + offs).to(tl.float32)
    lb = tl.load(LB + offs).to(tl.float32)
    o = d * inv * lg + lb

    tl.store(OUT + base + offs, o.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        _fused_post_kernel[(m,)](
            y, self.b1, self.rms2_w, self.b4, self.ln5_g, self.ln5_b, out,
            N=n, BLOCK=n,
            num_warps=4,
        )
        return out
