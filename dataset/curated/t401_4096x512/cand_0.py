import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 401
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _bias_triple_ln_kernel(
    X_ptr, OUT_ptr,
    B2_ptr,
    G3_ptr, B3_ptr,
    G4_ptr, B4_ptr,
    G5_ptr, B5_ptr,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # bias add (bf16 rounding to match reference)
    x = x + b2
    x = x.to(tl.bfloat16).to(tl.float32)

    inv_n = 1.0 / N

    # ---- LN 3 ----
    g = tl.load(G3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) * inv_n
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) * inv_n
    rstd = 1.0 / tl.sqrt(var + eps)
    x = d * rstd * g + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LN 4 ----
    g = tl.load(G4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) * inv_n
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) * inv_n
    rstd = 1.0 / tl.sqrt(var + eps)
    x = d * rstd * g + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LN 5 ----
    g = tl.load(G5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B5_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) * inv_n
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) * inv_n
    rstd = 1.0 / tl.sqrt(var + eps)
    y = d * rstd * g + b

    tl.store(OUT_ptr + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # two cuBLAS matmuls (tensor-core bf16, fp32 accumulate) as in reference
        x = x @ self.W0
        x = x @ self.W1
        x = x.contiguous()

        Mrows, N = x.shape
        out = torch.empty_like(x)
        _bias_triple_ln_kernel[(Mrows,)](
            x, out,
            self.b2,
            self.ln3_g, self.ln3_b,
            self.ln4_g, self.ln4_b,
            self.ln5_g, self.ln5_b,
            N, 1e-5,
            BLOCK=1024,
            num_warps=8,
        )
        return out
