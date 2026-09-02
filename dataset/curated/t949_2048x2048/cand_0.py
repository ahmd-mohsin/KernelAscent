import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 949
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _softmax_relu_gelu_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulate, matching PyTorch's internal upcast for bf16)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    denom = tl.sum(e, axis=0)
    s = e / denom

    # round to bf16 (softmax output dtype), relu is identity on softmax output
    s_bf16 = s.to(tl.bfloat16)
    s32 = s_bf16.to(tl.float32)
    s32 = tl.maximum(s32, 0.0)

    # exact GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = s32 * 0.5 * (1.0 + tl.math.erf(s32 * INV_SQRT2))

    tl.store(Y + row * stride_ym + cols, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            x = torch.relu(x)
            x = torch.relu(x)
            return F.gelu(x)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _softmax_relu_gelu_kernel[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
