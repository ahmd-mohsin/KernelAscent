import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 800
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _bias_rmsnorm_kernel(
    X, B, W, Y,
    N, stride_x, stride_y,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # bf16
    b = tl.load(B + cols, mask=mask, other=0.0)                    # bf16

    # bias add in bf16 (matches reference: bf16 + bf16 -> bf16)
    xb = (x.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)

    xf = xb.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)

    # (xf * rstd) rounded to bf16, then bf16-mul with w (fp32 math, round to bf16)
    a = (xf * rstd).to(tl.bfloat16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (a.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _bias_rmsnorm_kernel[(Mrows,)](
            x, self.b1, self.rms2_w, y,
            N, x.stride(0), y.stride(0),
            1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
