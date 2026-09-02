import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 866
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_gelu2_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    x = x.to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # first GELU (exact erf), round to bf16 to match reference dtype behavior
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # second GELU
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.bfloat16).to(tl.float32)

    # softmax in fp32
    x = tl.where(mask, x, float('-inf'))
    x_max = tl.max(x, axis=0)
    e = tl.exp(x - x_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = e / denom

    tl.store(Y + row * stride_ym + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = F.gelu(x)
            x = F.gelu(x)
            return torch.softmax(x, dim=-1)

        x = x.contiguous()
        m, n = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2d = x.view(-1, n)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _fused_gelu2_softmax_kernel[(m,)](
            x2d, y,
            x2d.stride(0), y.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(x.shape)
