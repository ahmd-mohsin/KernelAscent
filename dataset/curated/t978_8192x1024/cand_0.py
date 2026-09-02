import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 978
M, D, DT = 8192, 1024, torch.bfloat16


@triton.jit
def _fused_rms_ln_gelu_kernel(
    X, W_RMS, G, B, Y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x_ptr = X + row * N + cols
    x_bf = tl.load(x_ptr, mask=mask, other=0.0)
    xf = x_bf.to(tl.float32)

    # ---- RMSNorm (computed in fp32, then cast to bf16 like reference) ----
    ms = tl.sum(xf * xf, axis=0) / N
    inv_rms = 1.0 / tl.sqrt(ms + 1e-6)
    x_rms_bf = (xf * inv_rms).to(tl.bfloat16)

    w = tl.load(W_RMS + cols, mask=mask, other=0.0)
    # bf16 * bf16 multiply, result rounded to bf16 (matches reference)
    x1_bf = (x_rms_bf * w).to(tl.bfloat16)
    x1 = x1_bf.to(tl.float32)

    # ---- LayerNorm (fp32 internal math, like PyTorch) ----
    x1m = tl.where(mask, x1, 0.0)
    mean = tl.sum(x1m, axis=0) / N
    diff = tl.where(mask, x1 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (x1 - mean) * inv_std * g + b
    # PyTorch layer_norm outputs bf16, then gelu upcasts to fp32 internally
    y = y.to(tl.bfloat16).to(tl.float32)

    # ---- GELU (exact, erf-based, fp32 math) ----
    out = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))

    tl.store(Y + row * N + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda and x.dtype == torch.bfloat16
        x = x.contiguous()
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_ln_gelu_kernel[(M_,)](
            x, self.rms0_w, self.ln1_g, self.ln1_b, y,
            N=N, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
