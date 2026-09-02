import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 737
M, D, DT = 512, 1025, torch.float16


@triton.jit
def _fused_rms_kernel(
    X, W, B1, B2, Y,
    stride_x, stride_y,
    D_: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)

    # cast to fp16 exactly as reference: (_xf * rsqrt).to(half)
    h = (x * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0)

    # each op: fp32 internal math, rounded to fp16 (matches PyTorch half opmath)
    h = (h.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    h = (h.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)
    h = (h.to(tl.float32) + b2.to(tl.float32)).to(tl.float16)
    h = (h.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y + row * stride_y + cols, h, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x + self.b1
            x = x + self.b2
            return x * 1.2939

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_rms_kernel[(rows,)](
            x2, self.rms0_w, self.b1, self.b2, y,
            x2.stride(0), y.stride(0),
            D_=d,
            SCALE=1.2939,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
