import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 410
M, D, DT = 1024, 2049, torch.bfloat16


@triton.jit
def _fused_gelu_relu_softmax_kernel(
    X, Y,
    N,
    stride_xm, stride_ym,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact GELU (erf), computed in fp32 like PyTorch's opmath, then cast to bf16
    inv_sqrt2 = 0.7071067811865476
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * inv_sqrt2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # ReLU
    g = tl.maximum(g, 0.0)

    # Softmax in fp32
    g = tl.where(mask, g, float("-inf"))
    row_max = tl.max(g, axis=0)
    e = tl.exp(g - row_max)
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
            x = torch.relu(x)
            return torch.softmax(x, dim=-1)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)

        BLOCK_SIZE = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK_SIZE >= 2048 else 4
        _fused_gelu_relu_softmax_kernel[(m,)](
            x2, y, n,
            x2.stride(0), y.stride(0),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
