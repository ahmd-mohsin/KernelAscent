import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 502
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _fused_ln_rms_kernel(
    X, G, B, W, Y,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)

    # --- LayerNorm (fp32 math, matching PyTorch's mixed-precision path) ---
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * inv * g + b
    y_bf = y.to(tl.bfloat16)  # round to bf16 like layer_norm output

    # --- RMSNorm on the bf16 layernorm output (fp32 accumulation) ---
    yf = y_bf.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms = tl.sum(yf * yf, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)

    z = (yf * r).to(tl.bfloat16)  # round: (_xf * rsqrt(...)).to(bf16)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    z2 = (z.to(tl.float32) * w).to(tl.bfloat16)  # bf16 * bf16 -> bf16

    out = (z2.to(tl.float32) * SCALE).to(tl.bfloat16)  # x * 1.1595 -> bf16

    tl.store(Y + row * D + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_ln_rms_kernel[(m,)](
            x2, self.ln0_g, self.ln0_b, self.rms1_w, y,
            D=d,
            SCALE=1.1595,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
