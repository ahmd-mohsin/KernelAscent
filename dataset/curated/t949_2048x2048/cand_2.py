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
    stride_x, stride_y,
    N,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulation, matching PyTorch's bf16 softmax internals)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p = e / s

    # PyTorch materializes softmax output in bf16 before gelu -> replicate rounding
    p = p.to(Y.dtype.element_ty).to(tl.float32)

    # relu(relu(p)) is identity since p >= 0

    # exact GELU: 0.5 * p * (1 + erf(p / sqrt(2)))
    g = 0.5 * p * (1.0 + tl.math.erf(p * 0.7071067811865476))

    tl.store(Y + row * stride_y + offs, g.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        pass

    def forward(self, x):
        if not x.is_cuda:
            y = torch.softmax(x, dim=-1)
            y = torch.relu(y)
            y = torch.relu(y)
            return F.gelu(y)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.reshape(-1, N)
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()
        rows = x2d.shape[0]

        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4
        if BLOCK >= 2048:
            num_warps = 8
        if BLOCK >= 8192:
            num_warps = 16

        _softmax_gelu_kernel[(rows,)](
            x2d, y,
            x2d.stride(0), y.stride(0),
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.reshape(orig_shape)
