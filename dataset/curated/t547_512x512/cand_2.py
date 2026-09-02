import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 547
M, D, DT = 512, 512, torch.bfloat16


@triton.jit
def _fused_gelu_ln_rms_kernel(
    X_ptr, B2_ptr, G_ptr, B_ptr, B4_ptr, RW_ptr, Out_ptr,
    N, LN_EPS, RMS_EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf), computed in fp32 then rounded to bf16 (matches PyTorch)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # + b2 (elementwise add: fp32 opmath, bf16 result)
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    t = (g + b2).to(tl.bfloat16).to(tl.float32)

    # LayerNorm (stats in fp32, output rounded to bf16)
    mean = tl.sum(t, axis=0) / N
    d = tl.where(mask, t - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    gamma = tl.load(G_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (d * rstd * gamma + beta).to(tl.bfloat16).to(tl.float32)

    # + b4
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y + b4).to(tl.bfloat16).to(tl.float32)

    # RMSNorm: fp32 accumulate, round to bf16, then multiply by weight (bf16 op)
    ms = tl.sum(y * y, axis=0) / N
    r = tl.math.rsqrt(ms + RMS_EPS)
    z = (y * r).to(tl.bfloat16).to(tl.float32)
    rw = tl.load(RW_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (z * rw).to(tl.bfloat16)

    tl.store(Out_ptr + row * N + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # tensor-core bf16 matmul
        h = h.contiguous()
        Mrows, N = h.shape[0], h.shape[1]
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_ln_rms_kernel[(Mrows,)](
            h, self.b2, self.ln3_g, self.ln3_b, self.b4, self.rms5_w, out,
            N, 1e-5, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
