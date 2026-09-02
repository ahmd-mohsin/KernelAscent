import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 146
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_relu_scale_rmsnorm(
    X, W, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # relu (fp16) -> * 1.4873 computed in fp32 (PyTorch opmath), rounded to fp16
    xf = tl.maximum(x.to(tl.float32), 0.0)
    xf = xf * 1.4873
    x16 = xf.to(tl.float16)          # matches x.dtype rounding in reference
    xf = x16.to(tl.float32)          # _xf = x.float()

    # RMS norm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)
    normed = (xf * inv).to(tl.float16)

    # multiply by weight in fp16 (matches fp16 * fp16 in PyTorch)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = normed * w

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        n = orig_shape[-1]
        x2d = x.contiguous().view(-1, n)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n)
        _fused_relu_scale_rmsnorm[(m,)](
            x2d, self.rms2_w, y,
            x2d.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
