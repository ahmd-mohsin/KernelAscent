import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 888
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_gelu3_rmsnorm_kernel(
    X, W, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.352 (PyTorch computes bf16 scalar mul in fp32, rounds to bf16)
    x = (x * 1.352).to(tl.bfloat16).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # gelu #1 (exact erf gelu, computed in fp32, rounded to bf16 like PyTorch)
    x = (0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    # gelu #2
    x = (0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)
    # gelu #3
    x = (0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / D
    r = tl.math.rsqrt(ms + 1e-6)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16)

    tl.store(Y + row * D + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        d = x.shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(d)
        _fused_gelu3_rmsnorm_kernel[(m,)](
            x2d, self.rms4_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(x.shape)
