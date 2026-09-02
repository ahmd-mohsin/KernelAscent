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
    stride_xm, stride_ym,
    N, EPS, SCALE,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS)

    # (xf * rsqrt).to(fp16)  -- matches reference rounding
    n = (xf * r).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # fp16 * fp16 in PyTorch uses fp32 opmath, then rounds to fp16
    y = (n.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    # scalar mul also fp32 opmath, then rounds to fp16
    y = (y.to(tl.float32) * SCALE).to(tl.float16)

    tl.store(Y + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2d.shape
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 4096 else 4
        _rmsnorm_scale_kernel[(m,)](
            x2d, self.rms0_w, y,
            x2d.stride(0), y.stride(0),
            n, 1e-6, 1.0862,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
