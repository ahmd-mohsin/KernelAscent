import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 375
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _softmax_scale_gelu_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch's bf16 softmax path)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    s = e / denom

    # round to bf16 like the reference softmax output
    s = s.to(tl.bfloat16).to(tl.float32)

    # scale, round to bf16
    s = (s * SCALE).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = s * 0.5 * (1.0 + tl.math.erf(s * INV_SQRT2))

    tl.store(Y + row * stride_ym + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            y = torch.softmax(x, dim=-1)
            y = y * 1.3079
            return F.gelu(y)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _softmax_scale_gelu_kernel[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            n,
            SCALE=1.3079,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
