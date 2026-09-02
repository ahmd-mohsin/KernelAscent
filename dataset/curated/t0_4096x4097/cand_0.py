import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 0
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (exact, erf-based) computed in fp32, rounded to bf16 (matches PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # scale, round to bf16
    z = (g * 1.2523).to(tl.bfloat16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, z * z, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    v = (z * inv).to(tl.bfloat16).to(tl.float32)
    v = (v * w).to(tl.bfloat16).to(tl.float32)

    # softmax #1 (fp32 accumulate, bf16 round-trip like reference)
    v = tl.where(mask, v, float('-inf'))
    m1 = tl.max(v, axis=0)
    e1 = tl.where(mask, tl.exp(v - m1), 0.0)
    s1 = tl.sum(e1, axis=0)
    p1 = (e1 / s1).to(tl.bfloat16).to(tl.float32)

    # softmax #2
    p1 = tl.where(mask, p1, float('-inf'))
    m2 = tl.max(p1, axis=0)
    e2 = tl.where(mask, tl.exp(p1 - m2), 0.0)
    s2 = tl.sum(e2, axis=0)
    p2 = (e2 / s2).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_y + cols, p2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows, N = x.shape[0], x.shape[-1]
        x2d = x.view(-1, N)
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        _fused_row_kernel[(x2d.shape[0],)](
            x2d, self.rms2_w, y,
            N, x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view_as(x)
