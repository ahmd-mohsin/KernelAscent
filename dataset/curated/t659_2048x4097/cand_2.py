import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 659
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _relu_rmsnorm_kernel(
    X, W, Y,
    D: tl.constexpr,
    stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu (applied twice == once)
    x = tl.maximum(x, 0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + eps)

    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (xf * inv).to(tl.float16) * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            _xf = x.float()
            return (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w

        x = x.contiguous()
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 4096 else 4

        _relu_rmsnorm_kernel[(m,)](
            x2, self.rms2_w, y,
            d,
            x2.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
