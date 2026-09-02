import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 820
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _gelu_rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_x, stride_y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # round to bf16 to match reference (gelu output stored in bf16 before upcast)
    g_bf16 = g.to(tl.bfloat16)
    gf = g_bf16.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + EPS)
    normed = (gf * inv).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = normed * w

    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mr, Dr = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dr)
        _gelu_rmsnorm_kernel[(Mr,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            D=Dr, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
