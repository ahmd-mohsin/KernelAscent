import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 425
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_scale_rmsnorm_kernel(
    X, W, B, Y,
    D_dim,
    stride_x, stride_y,
    eps,
    scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_dim

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    # x * 1.3008 : computed in fp32 then rounded back to fp16 (matches PyTorch semantics)
    xs32 = x.to(tl.float32) * scale
    xs16 = xs32.to(tl.float16)

    xf = xs16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D_dim
    rrms = tl.math.rsqrt(ms + eps)

    normed = (xf * rrms).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B + cols, mask=mask, other=0.0)  # fp16

    y = normed * w + b
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        assert x.is_cuda
        x = x.contiguous()
        rows, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_scale_rmsnorm_kernel[(rows,)](
            x, self.rms1_w, self.b2, y,
            d,
            x.stride(0), y.stride(0),
            1e-6,
            1.3008,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
