import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 957
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _rmsnorm_kernel(
    X, W, Y,
    stride_x, stride_y,
    N,
    eps,
    HAS_SCALE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    if HAS_SCALE:
        s = tl.full((1,), SCALE, dtype=tl.float16)
        x = (x * s).to(tl.float16)

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * rstd).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)  # fp16
    y = xn * w
    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # x: (M, 1024) fp16
        x = x @ self.W0  # (M, 512)

        m, n = x.shape
        BLOCK = triton.next_power_of_2(n)

        y1 = torch.empty_like(x)
        _rmsnorm_kernel[(m,)](
            x, self.rms1_w, y1,
            x.stride(0), y1.stride(0),
            n, 1e-6,
            HAS_SCALE=False, SCALE=1.0,
            BLOCK=BLOCK,
            num_warps=4,
        )

        x2 = y1 @ self.W2  # (M, 512)

        y2 = torch.empty_like(x2)
        _rmsnorm_kernel[(m,)](
            x2, self.rms4_w, y2,
            x2.stride(0), y2.stride(0),
            n, 1e-6,
            HAS_SCALE=True, SCALE=1.2112,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y2
