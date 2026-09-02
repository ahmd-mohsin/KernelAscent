import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 842
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_bias_rmsnorm_scale_kernel(
    X, B, W, Y,
    D, stride_x, stride_y,
    C1, C2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = X + row * stride_x
    y_ptr = Y + row * stride_y

    # Pass 1: accumulate sum of squares of (x + b) in fp32
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, D, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        b = tl.load(B + offs, mask=mask, other=0.0)
        # fp16 add (exact in fp32, rounded to fp16 like the reference)
        s16 = (x.to(tl.float32) + b.to(tl.float32)).to(tl.float16)
        sf = s16.to(tl.float32)
        acc += tl.where(mask, sf * sf, 0.0)

    ssum = tl.sum(acc, axis=0)
    rstd = tl.math.rsqrt(ssum / D + 1e-6)

    # Pass 2: normalize, apply weight and constant scales with fp16 rounding
    for start in range(0, D, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < D
        x = tl.load(x_ptr + offs, mask=mask, other=0.0)
        b = tl.load(B + offs, mask=mask, other=0.0)
        w = tl.load(W + offs, mask=mask, other=0.0)

        s16 = (x.to(tl.float32) + b.to(tl.float32)).to(tl.float16)
        y = (s16.to(tl.float32) * rstd).to(tl.float16)
        y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
        y = (y.to(tl.float32) * C1).to(tl.float16)
        y = (y.to(tl.float32) * C2).to(tl.float16)

        tl.store(y_ptr + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference implementation
            x = x + self.b0
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = x * 1.3737
            x = x * 1.2539
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = 2048
        _fused_bias_rmsnorm_scale_kernel[(m,)](
            x2, self.b0, self.rms1_w, y,
            d, x2.stride(0), y.stride(0),
            1.3737, 1.2539,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
