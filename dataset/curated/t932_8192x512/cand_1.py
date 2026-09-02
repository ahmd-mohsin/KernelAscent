import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 932
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _fused_gelu_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # x = x * 1.069 (bf16 rounding to match reference)
    xf = (xf * 1.069).to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))), rounded to bf16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(g * g, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    normed = (g * inv).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (normed * w).to(tl.bfloat16)

    tl.store(Out_ptr + row * stride_xm + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_gelu_rms_kernel[(m,)](
            x, self.rms3_w, out,
            x.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
