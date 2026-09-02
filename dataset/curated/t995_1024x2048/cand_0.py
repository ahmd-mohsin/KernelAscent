import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 995
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_ln_gelu2_rms_relu(
    X, OUT, G, B, W,
    N,
    stride_x, stride_o,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm (fp32 math, like PyTorch half layer_norm) ----
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + 1e-5)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b
    # round to fp16 (layer_norm output dtype)
    y = y.to(tl.float16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476
    # ---- GELU #1 (erf, fp32 opmath, round to fp16) ----
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.float16).to(tl.float32)

    # ---- GELU #2 ----
    y = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))
    y = y.to(tl.float16).to(tl.float32)

    # ---- RMSNorm (fp32 mean of squares, rsqrt, cast to fp16, fp16 mul by weight) ----
    y_masked = tl.where(mask, y, 0.0)
    ms = tl.sum(y_masked * y_masked, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    yh = (y * r).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0)  # fp16
    z = yh * w  # fp16 multiply, matching torch half * half

    # ---- ReLU ----
    zero = tl.zeros_like(z)
    z = tl.maximum(z, zero)

    tl.store(OUT + row * stride_o + offs, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback: reference path
            x = x @ self.W0
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = F.gelu(x)
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return torch.relu(x)

        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()

        rows, N = h.shape[0], h.shape[1]
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_gelu2_rms_relu[(rows,)](
            h, out, self.ln1_g, self.ln1_b, self.rms4_w,
            N,
            h.stride(0), out.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
