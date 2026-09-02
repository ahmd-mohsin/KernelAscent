import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 789
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_scale_relu_softmax(
    X, Y,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # multiply in fp32 (opmath), round to bf16 like PyTorch elementwise op
    xf = x.to(tl.float32) * SCALE
    xb = xf.to(tl.bfloat16)
    # relu (idempotent, one is enough)
    xb = tl.maximum(xb, 0.0)

    # softmax in fp32 (matches PyTorch's fp32 accumulation for bf16)
    z = xb.to(tl.float32)
    z = tl.where(mask, z, float('-inf'))
    m = tl.max(z, axis=0)
    e = tl.exp(z - m)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.3032
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK_N <= 1024 else 8
        _fused_scale_relu_softmax[(m,)](
            x2, y,
            x2.stride(0), y.stride(0),
            n,
            SCALE=1.3032,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
