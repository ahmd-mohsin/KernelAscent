import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 359
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_norms_kernel(
    X, Y,
    LN0G, LN0B, RMS2W, LN3G, LN3B,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (eps=1e-5), computed in fp32, rounded to bf16 ----
    mean0 = tl.sum(x, axis=0) / N
    d0 = tl.where(mask, x - mean0, 0.0)
    var0 = tl.sum(d0 * d0, axis=0) / N
    rstd0 = 1.0 / tl.sqrt(var0 + 1e-5)
    g0 = tl.load(LN0G + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(LN0B + cols, mask=mask, other=0.0).to(tl.float32)
    x = (d0 * rstd0) * g0 + b0
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- scale by 1.0166 (bf16 result rounding) ----
    x = (x * 1.0166).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm (fp32 math, eps=1e-6), round to bf16, then * rms2_w ----
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / N
    x = x * (1.0 / tl.sqrt(ms + 1e-6))
    x = x.to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(RMS2W + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x * w2).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm 3 (eps=1e-5) ----
    mean3 = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d3 = tl.where(mask, x - mean3, 0.0)
    var3 = tl.sum(d3 * d3, axis=0) / N
    rstd3 = 1.0 / tl.sqrt(var3 + 1e-5)
    g3 = tl.load(LN3G + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(LN3B + cols, mask=mask, other=0.0).to(tl.float32)
    x = (d3 * rstd3) * g3 + b3
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- scale by 1.0289 ----
    y = (x * 1.0289).to(tl.bfloat16)

    tl.store(Y + row * N + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_norms_kernel[(rows,)](
            x2, y,
            self.ln0_g, self.ln0_b, self.rms2_w, self.ln3_g, self.ln3_b,
            N, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
