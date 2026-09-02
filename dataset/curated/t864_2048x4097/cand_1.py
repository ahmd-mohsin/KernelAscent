import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 864
M, D, DT = 2048, 4097, torch.float16


@triton.jit
def _fused_relu_rms_relu(X, W, Y, N, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu, then compute in fp32
    xf = tl.maximum(x, 0.0).to(tl.float32)

    mean = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(mean + eps)

    # cast normalized value back to fp16 BEFORE multiplying by w (matches reference)
    xn = (xf * inv).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = xn * w
    y = tl.maximum(y, 0.0)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


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
        x2 = x.contiguous().view(-1, orig_shape[-1])
        Mrows, N = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        _fused_relu_rms_relu[(Mrows,)](
            x2, self.rms1_w, y,
            N, x2.stride(0), y.stride(0),
            1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
