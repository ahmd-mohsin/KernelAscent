import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 46
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_add_ln_add(
    X, B0, G, B, B2, Y,
    N, eps,
    stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y

    # Pass 1: compute mean and variance of (x + b0) in fp32
    _sum = tl.zeros([BLOCK], dtype=tl.float32)
    _sq = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
        v = tl.where(mask, x + b0, 0.0)
        _sum += v
        _sq += v * v
    mean = tl.sum(_sum, axis=0) / N
    var = tl.sum(_sq, axis=0) / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize, affine, add b2
    for off in range(0, N, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
        v = (x + b0 - mean) * rstd
        out = v * g + b + b2
        tl.store(y_ptr + cols, out.to(Y.dtype.element_ty), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = x + self.b0
            y = F.layer_norm(y, (y.shape[-1],), self.ln1_g, self.ln1_b)
            return y + self.b2
        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = 2048
        _fused_add_ln_add[(rows,)](
            x2, self.b0, self.ln1_g, self.ln1_b, self.b2, y,
            N, 1e-5,
            x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
