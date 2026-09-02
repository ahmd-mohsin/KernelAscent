import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 906
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _gelu_relu_softmax_kernel(
    X, Y,
    stride_xm, stride_ym,
    N, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # GELU (erf variant), computed in fp32 then rounded to bf16 to match
    # the reference (which applies gelu on a bf16 tensor)
    g = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))
    # ReLU
    g = tl.maximum(g, 0.0)
    # round to bf16 like the reference intermediate, then upcast for softmax
    g = g.to(tl.bfloat16).to(tl.float32)

    # Softmax (fp32 accumulation, matches PyTorch's bf16 softmax behavior)
    g = tl.where(mask, g, float('-inf'))
    m = tl.max(g, axis=0)
    e = tl.exp(g - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = e / s

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
        Mrows, N = x2.shape
        y = torch.empty_like(x2)
        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK_N >= 2048:
            num_warps = 8
        if BLOCK_N >= 8192:
            num_warps = 16
        _gelu_relu_softmax_kernel[(Mrows,)](
            x2, y,
            x2.stride(0), y.stride(0),
            N, BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
