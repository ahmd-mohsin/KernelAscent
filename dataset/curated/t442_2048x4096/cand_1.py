import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 442
M, D, DT = 2048, 4096, torch.bfloat16


@triton.jit
def _fused_norm_chain(
    X, Y,
    RW1, G2, B2, RW3, G4, B4,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    ptr = X + row * N + offs

    x = tl.load(ptr).to(tl.float32)

    # ---- RMSNorm 1 (eps=1e-6), cast to bf16 then multiply by weight ----
    ms = tl.sum(x * x, axis=0) / N
    r = tl.rsqrt(ms + 1e-6)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(RW1 + offs).to(tl.float32)
    x = (x * w1).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 2 (eps=1e-5) ----
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)
    g2 = tl.load(G2 + offs).to(tl.float32)
    b2 = tl.load(B2 + offs).to(tl.float32)
    x = (xc * rstd * g2 + b2).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm 3 (eps=1e-6) ----
    ms = tl.sum(x * x, axis=0) / N
    r = tl.rsqrt(ms + 1e-6)
    x = (x * r).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(RW3 + offs).to(tl.float32)
    x = (x * w3).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 4 (eps=1e-5) ----
    mean = tl.sum(x, axis=0) / N
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)
    g4 = tl.load(G4 + offs).to(tl.float32)
    b4 = tl.load(B4 + offs).to(tl.float32)
    out = (xc * rstd * g4 + b4).to(tl.bfloat16)

    tl.store(Y + row * N + offs, out)


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
        x = x @ self.W0  # (M, 1024), cuBLAS bf16 tensor-core GEMM
        x = x.contiguous()
        rows, n = x.shape
        y = torch.empty_like(x)
        _fused_norm_chain[(rows,)](
            x, y,
            self.rms1_w, self.ln2_g, self.ln2_b,
            self.rms3_w, self.ln4_g, self.ln4_b,
            N=n, BLOCK=n,
            num_warps=8,
        )
        return y @ self.W5
