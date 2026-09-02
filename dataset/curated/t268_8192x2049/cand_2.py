import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 268
M, D, DT = 8192, 2049, torch.bfloat16


@triton.jit
def _rms_scale_kernel(
    X, W, Y,
    D, stride_x, stride_y,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    ms = tl.sum(x * x, axis=0) / D
    rs = tl.math.rsqrt(ms + 1e-6)

    # normalized value rounded to bf16 (matches .to(x.dtype))
    xn = (x * rs).to(tl.bfloat16).to(tl.float32)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)

    # bf16 * bf16 multiply (fp32 opmath, rounded to bf16)
    y = (xn * w).to(tl.bfloat16).to(tl.float32)

    # bf16 tensor * python-float scalar (fp32 opmath, rounded to bf16 on store)
    y = y * SCALE

    tl.store(Y + row * stride_y + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            _xf = x.float()
            out = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            return out * 1.4423

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]

        y = torch.empty_like(x2)
        w = self.rms0_w
        if not w.is_cuda:
            w = w.to(x.device)
        w = w.contiguous()

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _rms_scale_kernel[(m,)](
            x2, w, y,
            d, x2.stride(0), y.stride(0),
            SCALE=1.4423,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
