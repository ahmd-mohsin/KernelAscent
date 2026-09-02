import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 204
M, D, DT = 4096, 512, torch.float16


@triton.jit
def _fused_act_softmax_kernel(
    X_ptr, Out_ptr,
    N,
    stride_xm, stride_om,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=float('-inf'))

    # relu (in fp16, same as reference)
    x = tl.maximum(x, 0.0)

    # gelu (erf-based, exact), compute in fp32 then round back to fp16
    xf = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g = g.to(tl.float16)

    # scale, round to fp16
    y = (g * SCALE).to(tl.float16)

    # softmax with fp32 accumulation
    yf = tl.where(mask, y.to(tl.float32), float('-inf'))
    row_max = tl.max(yf, axis=0)
    e = tl.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Out_ptr + row * stride_om + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        M_, N_ = x.shape
        out = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N_)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _fused_act_softmax_kernel[(M_,)](
            x, out,
            N_,
            x.stride(0), out.stride(0),
            SCALE=1.2304,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return out
