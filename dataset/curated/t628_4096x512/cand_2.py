import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 628
M, D, DT = 4096, 512, torch.float16

_INV_SQRT2 = 0.7071067811865476


@triton.jit
def _fused_softmax_gelu3_softmax(
    X_ptr, Y_ptr,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax 1 (fp32 accumulation, matches PyTorch half softmax)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    p = e / s
    # round to fp16 (op boundary)
    p = p.to(tl.float16).to(tl.float32)

    # 3x exact GELU (erf-based), fp32 math with fp16 rounding at op boundaries
    inv_sqrt2 = 0.7071067811865476
    p = 0.5 * p * (1.0 + tl.math.erf(p * inv_sqrt2))
    p = p.to(tl.float16).to(tl.float32)
    p = 0.5 * p * (1.0 + tl.math.erf(p * inv_sqrt2))
    p = p.to(tl.float16).to(tl.float32)
    p = 0.5 * p * (1.0 + tl.math.erf(p * inv_sqrt2))
    p = p.to(tl.float16).to(tl.float32)

    # softmax 2
    p = tl.where(mask, p, float('-inf'))
    m2 = tl.max(p, axis=0)
    e2 = tl.exp(p - m2)
    s2 = tl.sum(tl.where(mask, e2, 0.0), axis=0)
    out = e2 / s2

    tl.store(Y_ptr + row * stride_y + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            x = F.gelu(x)
            x = F.gelu(x)
            x = torch.softmax(x, dim=-1)
            return x

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        rows, N = x2.shape
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _fused_softmax_gelu3_softmax[(rows,)](
            x2, y,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
