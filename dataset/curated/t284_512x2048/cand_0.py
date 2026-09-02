import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 284
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_gelu_scale_bias_rms_kernel(
    X, B2, W, OUT,
    n_cols,
    stride_x, stride_o,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU (erf-based), rounded to bf16 like PyTorch elementwise op
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale, round to bf16
    g = (g * 1.164).to(tl.bfloat16).to(tl.float32)

    # bias add, round to bf16
    b = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    g = (g + b).to(tl.bfloat16).to(tl.float32)

    # RMS norm in fp32
    ssq = tl.sum(tl.where(mask, g * g, 0.0), axis=0)
    rs = tl.math.rsqrt(ssq / n_cols + eps)
    y = (g * rs).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b2 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_gelu_scale_bias_rms_kernel[(n_rows,)](
            x2d, self.b2, self.rms3_w, out,
            n_cols,
            x2d.stride(0), out.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
