import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 704
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_ln_gelu_rms_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    N,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ReLU
    x = tl.maximum(x, 0.0)

    # LayerNorm (fp32 accumulation, like PyTorch's bf16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.math.rsqrt(var + eps_ln)

    g = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * g + b
    # round to bf16 like PyTorch layer_norm output
    y = y.to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), computed in fp32 like PyTorch opmath, then rounded to bf16
    ge = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
    ge = ge.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32 (matches explicit .float() in reference)
    ge_m = tl.where(mask, ge, 0.0)
    ms = tl.sum(ge_m * ge_m, axis=0) / N
    r = tl.math.rsqrt(ms + eps_rms)
    z = (ge * r).to(tl.bfloat16).to(tl.float32)

    # multiply by rms weight (bf16 rounding like reference)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (z * w).to(tl.bfloat16).to(tl.float32)

    # final scale, round to bf16
    out = (z * scale).to(tl.bfloat16)
    tl.store(Y_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 1024, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS bf16 matmul (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        _fused_ln_gelu_rms_kernel[(Mrows,)](
            h, self.ln2_g, self.ln2_b, self.rms4_w, out,
            N,
            1e-5, 1e-6, 1.1698,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
