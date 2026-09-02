import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 864
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _fused_relu_rms_relu_kernel(
    X, W, Y,
    D: tl.constexpr,
    stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y

    # Pass 1: accumulate sum of relu(x)^2 in fp32
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, D, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        x = tl.maximum(x, 0.0)
        acc += x * x
    mean = tl.sum(acc, axis=0) / D
    rstd = 1.0 / tl.sqrt(mean + eps)

    # Pass 2: normalize, scale by w, relu, store
    for off in range(0, D, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < D
        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        x = tl.maximum(x, 0.0)
        xn = (x * rstd).to(tl.float16)
        w = tl.load(W + idx, mask=mask, other=0.0)
        y = xn * w
        y = tl.maximum(y, 0.0)
        tl.store(y_ptr + idx, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return torch.relu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = 1024
        _fused_relu_rms_relu_kernel[(m,)](
            x2, self.rms1_w, y,
            d, x2.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
