import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 949
M, D, DT = 2048, 2048, torch.bfloat16


@triton.jit
def _softmax_gelu_kernel(
    X, Y,
    stride_xm, stride_ym,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch internals for bf16)
    row_max = tl.max(x, axis=0)
    e = tl.exp(x - row_max)
    denom = tl.sum(e, axis=0)
    s = e / denom

    # cast to bf16 (softmax output dtype), then relu is identity on softmax output
    s_bf16 = s.to(tl.bfloat16)

    # exact GELU (erf), computed in fp32 on the bf16-rounded value
    v = s_bf16.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = v * 0.5 * (1.0 + tl.math.erf(v * INV_SQRT2))

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

        x = x.contiguous()
        m, n = x.shape[-2] if x.dim() > 1 else 1, x.shape[-1]
        x2d = x.view(-1, n)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_N >= 2048 else 4
        _softmax_gelu_kernel[(rows,)](
            x2d, y,
            x2d.stride(0), y.stride(0),
            n,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(x.shape)
