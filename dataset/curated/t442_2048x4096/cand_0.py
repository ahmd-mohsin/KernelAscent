import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 442
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_norms_kernel(
    X, Y,
    RMS1_W, LN2_G, LN2_B, RMS3_W, LN4_G, LN4_B,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    x = tl.load(ptr).to(tl.float32)

    # ---- RMSNorm 1 (eps=1e-6), computed in fp32, rounded to bf16, scaled ----
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16)
    w1 = tl.load(RMS1_W + offs).to(tl.float32)
    x = (x.to(tl.float32) * w1).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 (eps=1e-5) ----
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    g2 = tl.load(LN2_G + offs).to(tl.float32)
    b2 = tl.load(LN2_B + offs).to(tl.float32)
    x = (xc * tl.math.rsqrt(var + 1e-5) * g2 + b2).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 3 (eps=1e-6) ----
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16)
    w3 = tl.load(RMS3_W + offs).to(tl.float32)
    x = (x.to(tl.float32) * w3).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 4 (eps=1e-5) ----
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    g4 = tl.load(LN4_G + offs).to(tl.float32)
    b4 = tl.load(LN4_B + offs).to(tl.float32)
    y = (xc * tl.math.rsqrt(var + 1e-5) * g4 + b4).to(tl.bfloat16)

    tl.store(Y + row * N + offs, y)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        x = x @ self.W0
        x = x.contiguous()
        rows, n = x.shape

        y = torch.empty_like(x)
        _fused_norms_kernel[(rows,)](
            x, y,
            self.rms1_w, self.ln2_g, self.ln2_b,
            self.rms3_w, self.ln4_g, self.ln4_b,
            N=n, BLOCK=n,
            num_warps=8,
        )

        # GEMM 2
        return y @ self.W5
