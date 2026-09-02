import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 732
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _rmsnorm_scale_kernel(
    X, W, Y,
    N,  # row length
    stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # mean of squares in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    # match reference: cast to fp16 after rsqrt-mul
    t1 = (xf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # half*half computed in fp32 (PyTorch opmath), rounded to half
    t2 = (t1.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # half * python-float scalar computed in fp32, rounded to half
    out = (t2.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            return x * 1.0862

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK_N = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK_N >= 4096 else 4
        _rmsnorm_scale_kernel[(rows,)](
            x2, self.rms0_w, y,
            N,
            x2.stride(0), y.stride(0),
            SCALE=1.0862,
            BLOCK_N=BLOCK_N,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
