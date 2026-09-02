import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 436
M, D, DT = 4096, 4096, torch.float16


@triton.jit
def _rmsnorm_scale_kernel(
    X_ptr, W_ptr, Y_ptr,
    stride_xm, stride_ym,
    N,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32) * SCALE

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    # match reference: cast normalized value to fp16, then multiply by fp16 weight
    norm_h = (xf * inv).to(tl.float16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    y = norm_h * w

    tl.store(Y_ptr + row * stride_ym + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4
        _rmsnorm_scale_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0), y.stride(0),
            n,
            SCALE=1.1783,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y @ self.W2
