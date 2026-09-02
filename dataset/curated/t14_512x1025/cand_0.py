import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 14
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_relu_rms_rms_relu(
    X, W1, W2, Y,
    D, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    # relu (in input dtype)
    x = tl.maximum(x, 0.0)

    # RMSNorm 1
    xf = x.to(tl.float32)
    ms1 = tl.sum(xf * xf, axis=0) / D
    r1 = tl.math.rsqrt(ms1 + 1e-6)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    x1 = (xf * r1).to(tl.bfloat16) * w1

    # RMSNorm 2
    xf1 = x1.to(tl.float32)
    ms2 = tl.sum(xf1 * xf1, axis=0) / D
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    x2 = (xf1 * r2).to(tl.bfloat16) * w2

    # final relu
    y = tl.maximum(x2, 0.0)
    tl.store(Y + row * stride_y + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return torch.relu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        y = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_relu_rms_rms_relu[(m,)](
            x2d, self.rms1_w, self.rms2_w, y,
            d, x2d.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
