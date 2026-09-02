import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 455
M, D, DT = 512, 4096, torch.float16


@triton.jit
def _rms_scale_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    # step 1: normalize in fp32, round to fp16 (matches .to(x.dtype))
    n_h = (x * rstd).to(tl.float16)

    # step 2: multiply by weight; PyTorch half*half computes in fp32, rounds to fp16
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    t2 = (n_h.to(tl.float32) * w).to(tl.float16)

    # step 3: multiply by scalar 1.1255; computed in fp32, rounded to fp16
    t3 = (t2.to(tl.float32) * 1.1255).to(tl.float16)

    tl.store(Y + row * stride_y + cols, t3, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_scale_kernel[(Mrows,)](
            x, self.rms1_w, y,
            N, x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
