import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 567
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_norm_act_kernel(
    X, W_RMS, LN_G, LN_B, OUT,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (computed in fp32, cast to bf16, then scaled)
    ms = tl.sum(x * x, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    y = (x * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_RMS + cols, mask=mask, other=0.0).to(tl.float32)
    y = (y * w).to(tl.bfloat16).to(tl.float32)

    # ReLU
    y = tl.maximum(y, 0.0)

    # LayerNorm (fp32 stats, eps=1e-5)
    mean = tl.sum(tl.where(mask, y, 0.0), axis=0) / N
    d = tl.where(mask, y - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((y - mean) * rstd * g + b).to(tl.bfloat16).to(tl.float32)

    # GELU (erf) twice, with bf16 rounding between (matches PyTorch opmath)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    y = (0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    y = (0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))).to(tl.bfloat16)

    tl.store(OUT + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_act_kernel[(Mrows,)](
            x, self.rms1_w, self.ln3_g, self.ln3_b, out,
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
