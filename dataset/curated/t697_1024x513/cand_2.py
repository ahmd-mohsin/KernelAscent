import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 697
M, D, DT = 1024, 513, torch.float16


@triton.jit
def _fused_triple_ln_kernel(
    X, OUT,
    G1, B1, B2, G3, B3, G4, B4,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row * N

    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 1 (compute in fp32, round to fp16 like PyTorch) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G1 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B1 + offs, mask=mask, other=0.0).to(tl.float32)
    y16 = (d * inv * g + b).to(tl.float16)

    # ---- add b2 (fp16 add, like PyTorch) ----
    b2 = tl.load(B2 + offs, mask=mask, other=0.0)
    y16 = y16 + b2

    # ---- LayerNorm 3 ----
    x = y16.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    y16 = (d * inv * g + b).to(tl.float16)

    # ---- LayerNorm 4 ----
    x = y16.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + eps)
    g = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y16 = (d * inv * g + b).to(tl.float16)

    tl.store(OUT + base + offs, y16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 1024, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        rows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_triple_ln_kernel[(rows,)](
            h, out,
            self.ln1_g, self.ln1_b, self.b2,
            self.ln3_g, self.ln3_b,
            self.ln4_g, self.ln4_b,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
