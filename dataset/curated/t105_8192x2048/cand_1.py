import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

SEED = 105
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _double_rmsnorm_kernel(
    X, W1, W2, Out,
    stride_x, stride_o,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16
    xf = x.to(tl.float32)

    # First RMSNorm
    ms1 = tl.sum(xf * xf, axis=0) / D
    r1 = tl.math.rsqrt(ms1 + EPS)
    y_h = (xf * r1).to(tl.float16)  # cast to fp16 first (matches reference)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)  # fp16
    y_h = y_h * w1  # fp16 multiply, matches torch fp16 arithmetic

    # Second RMSNorm
    zf = y_h.to(tl.float32)
    ms2 = tl.sum(zf * zf, axis=0) / D
    r2 = tl.math.rsqrt(ms2 + EPS)
    z_h = (zf * r2).to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    z_h = z_h * w2

    tl.store(Out + row * stride_o + cols, z_h, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 GEMM (tensor cores)
        Mrows, Dcols = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        num_warps = 8 if BLOCK >= 2048 else 4
        _double_rmsnorm_kernel[(Mrows,)](
            x, self.rms1_w, self.rms2_w, out,
            x.stride(0), out.stride(0),
            D=Dcols, EPS=1e-6, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
