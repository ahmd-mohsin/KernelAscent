import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 679
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _double_rmsnorm_kernel(
    X, W0, W1, Y,
    stride_x, stride_y,
    N: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # first RMSNorm
    ms0 = tl.sum(xf * xf, axis=0) / N
    r0 = 1.0 / tl.sqrt(ms0 + EPS)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    # cast to fp16 before multiplying by w0 (fp16 * fp16 -> fp16), matching reference
    x1 = (xf * r0).to(tl.float16) * w0

    # second RMSNorm
    x1f = x1.to(tl.float32)
    ms1 = tl.sum(x1f * x1f, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + EPS)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    y = (x1f * r1).to(tl.float16) * w1

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        x2d = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2d.shape
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 4096 else 4
        _double_rmsnorm_kernel[(m,)](
            x2d, self.rms0_w, self.rms1_w, y,
            x2d.stride(0), y.stride(0),
            N=n, EPS=1e-6, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
