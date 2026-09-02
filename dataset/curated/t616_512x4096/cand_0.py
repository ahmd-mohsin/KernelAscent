import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 616
M, D, DT = 512, 4096, torch.float16


@triton.jit
def fused_kernel(x_ptr, g_ptr, b_ptr, out_ptr,
                 N, eps,
                 BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based), computed in fp32 then cast to fp16 to match reference
    inv_sqrt2 = 0.7071067811865476
    y = 0.5 * x * (1.0 + tl.math.erf(x * inv_sqrt2))
    y = y.to(tl.float16).to(tl.float32)

    # Softmax (fp32 accumulation), cast to fp16 to match reference
    y_m = tl.where(mask, y, float('-inf'))
    row_max = tl.max(y_m, axis=0)
    num = tl.exp(y_m - row_max)
    num = tl.where(mask, num, 0.0)
    denom = tl.sum(num, axis=0)
    s = num / denom
    s = s.to(tl.float16).to(tl.float32)

    # ReLU (no-op on nonneg, kept for exactness)
    s = tl.maximum(s, 0.0)

    # LayerNorm in fp32
    mean = tl.sum(tl.where(mask, s, 0.0), axis=0) / N
    diff = tl.where(mask, s - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    g = tl.load(g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (s - mean) * rstd * g + b

    tl.store(out_ptr + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        fused_kernel[(Mrows,)](
            x, self.ln3_g, self.ln3_b, out,
            N, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
