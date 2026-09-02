import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 275
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_rms_ln_gelu(
    X, W, G, B, Y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- RMSNorm (fp32 math, round to bf16, mul by weight in bf16 semantics) ----
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    x = x * tl.math.rsqrt(ms + 1e-6)
    # torch: (.to(bf16)) * rms0_w  -> bf16 rounding after normalize, then bf16 mul
    x = x.to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation like PyTorch) ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    inv_std = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = xm * inv_std * g + b

    # ---- GELU (exact, erf-based, fp32 math like PyTorch opmath) ----
    y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * N + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            return F.gelu(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_rms_ln_gelu[(rows,)](
            x2, self.rms0_w, self.ln1_g, self.ln1_b, y,
            N, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
