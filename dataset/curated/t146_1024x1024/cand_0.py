import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 146
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_relu_scale_rmsnorm(X, W, Y, N, stride, eps, scale,
                              BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride + cols, mask=mask, other=0.0)
    # relu in fp16
    x = tl.maximum(x, 0.0)
    # scale in fp16 (match fp16 rounding of reference)
    x = (x.to(tl.float32) * scale).to(tl.float16)

    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)

    y = (xf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = y * w
    tl.store(Y + row * stride + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = x * 1.4873
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return x

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        Mrows, N = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 4 if BLOCK <= 2048 else 8
        _fused_relu_scale_rmsnorm[(Mrows,)](
            x2, self.rms2_w, y, N, x2.stride(0), 1e-6, 1.4873,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
