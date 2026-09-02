import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 766
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_rms_ln_gelu(
    X, W1, G, B, Y,
    N,
    eps_rms, eps_ln,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    out_ty = Y.dtype.element_ty

    # ---- RMSNorm (computed in fp32, rounded to bf16, then weighted) ----
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    xn = x * tl.math.rsqrt(ms + eps_rms)
    xn = xn.to(out_ty)  # round like `.to(x.dtype)` in reference

    w1 = tl.load(W1 + cols, mask=mask, other=0.0).to(tl.float32)
    h = (xn.to(tl.float32) * w1).to(out_ty)  # bf16*bf16 with fp32 opmath

    # ---- LayerNorm (fp32 internals like PyTorch) ----
    hf = h.to(tl.float32)
    mu = tl.sum(hf, axis=0) / N
    d = tl.where(mask, hf - mu, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + eps_ln)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    yln = (d * inv * g + b).to(out_ty)  # rounded to bf16 as layer_norm output

    # ---- GELU (erf, fp32 opmath) ----
    yf = yln.to(tl.float32)
    out = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y + row * N + cols, out.to(out_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # tensor-core GEMM via cuBLAS
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_ln_gelu[(Mrows,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, y,
            N, 1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
