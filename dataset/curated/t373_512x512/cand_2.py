import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 373
M, D, DT = 512, 512, torch.float16


@triton.jit
def _fused_gelu_ln_gelu(
    X, OUT, G, B,
    N, stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf) computed in fp32, rounded to fp16 to match PyTorch storage
    inv_sqrt2: tl.constexpr = 0.7071067811865476
    h = x * 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))
    h = h.to(tl.float16).to(tl.float32)

    # LayerNorm in fp32 (matches PyTorch internal fp32 accumulation)
    mean = tl.sum(tl.where(mask, h, 0.0), axis=0) / N
    d = tl.where(mask, h - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (h - mean) * rstd * g + b
    y = y.to(tl.float16).to(tl.float32)

    # Final GELU
    out = y * 0.5 * (1.0 + tl.math.erf(y * inv_sqrt2))
    tl.store(OUT + row * stride_o + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_gelu_ln_gelu[(Mrows,)](
            h, out, self.ln2_g, self.ln2_b,
            N, h.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out
