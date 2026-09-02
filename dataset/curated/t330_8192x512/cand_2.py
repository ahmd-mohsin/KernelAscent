import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 330
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, Y,
    N,
    eps_ln, eps_rms,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, like PyTorch's bf16 layer_norm)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = tl.math.rsqrt(var + eps_ln)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b

    # Round to bf16 (this intermediate rounding matches the reference)
    yb = y.to(tl.bfloat16)
    yf = yb.to(tl.float32)

    # RMSNorm in fp32 on the bf16 intermediate
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = tl.math.rsqrt(ms + eps_rms)
    z = (yf * r).to(tl.bfloat16)

    # bf16 * bf16 -> bf16 (exact product in fp32, then round: matches hw bf16 mul)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (z.to(tl.float32) * w).to(tl.bfloat16)

    tl.store(Y + row * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # Matmul via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_ln_rms_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N, 1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
