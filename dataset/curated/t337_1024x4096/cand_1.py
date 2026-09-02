import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 337
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _rms_relu_kernel(
    X, W, Y,
    stride_x, stride_y,
    D_: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / D_
    rstd = 1.0 / tl.sqrt(ms + EPS)

    # normalize in fp32, round to bf16 (matches .to(x.dtype))
    xn = (xf * rstd).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 * bf16 elementwise (PyTorch computes in fp32, rounds once)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # relu
    zero = tl.zeros_like(y)
    y = tl.where(y > 0, y, zero)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            return torch.relu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 4096 else 4

        _rms_relu_kernel[(m,)](
            x2, self.rms0_w, y,
            x2.stride(0), y.stride(0),
            d, 1e-6, BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
