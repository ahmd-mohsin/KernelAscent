import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 45
M, D, DT = 2048, 4096, torch.float16


@triton.jit
def _fused_bias_rms_kernel(
    X_ptr, B1_ptr, B2_ptr, W_ptr, B4_ptr, Y_ptr,
    N, stride_row,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_row + cols, mask=mask, other=0.0)
    b1 = tl.load(B1_ptr + cols, mask=mask, other=0.0)
    b2 = tl.load(B2_ptr + cols, mask=mask, other=0.0)

    # match reference order: (x + b1) + b2 in fp16
    x = x + b1
    x = x + b2

    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = tl.math.rsqrt(ms + EPS)

    xn = (xf * inv).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    b4 = tl.load(B4_ptr + cols, mask=mask, other=0.0)

    y = xn * w
    y = y + b4

    tl.store(Y_ptr + row * stride_row + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores (same op as reference)
        x = x @ self.W0
        x = x.contiguous()

        Mrows, N = x.shape
        y = torch.empty_like(x)

        BLOCK = triton.next_power_of_2(N)
        _fused_bias_rms_kernel[(Mrows,)](
            x, self.b1, self.b2, self.rms3_w, self.b4, y,
            N, x.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
