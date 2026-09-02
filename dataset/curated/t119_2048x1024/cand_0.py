import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 119
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _scale_rmsnorm_kernel(
    X, W, Y,
    N,
    stride_xm, stride_ym,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)

    # x = x * 1.4953 (computed in fp32, rounded to bf16 like PyTorch)
    xs_bf16 = (x.to(tl.float32) * SCALE).to(tl.bfloat16)

    # _xf = x.float()
    xf = xs_bf16.to(tl.float32)

    ms = tl.sum(xf * xf, axis=0) / N
    rstd = tl.math.rsqrt(ms + EPS)

    # (_xf * rstd).to(bf16) then * w (elementwise mul in fp32, rounded to bf16)
    y_bf16 = (xf * rstd).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (y_bf16.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 4096, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul
        M_, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _scale_rmsnorm_kernel[(M_,)](
            x, self.rms2_w, y,
            N,
            x.stride(0), y.stride(0),
            SCALE=1.4953,
            EPS=1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
