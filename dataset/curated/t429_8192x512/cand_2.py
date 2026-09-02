import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 429
M, D, DT = 8192, 512, torch.bfloat16


@triton.jit
def _rmsnorm_softmax_kernel(
    X_ptr, W_ptr, Y_ptr,
    N,
    stride_xm, stride_ym,
    EPS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm
    ms = tl.sum(xf * xf, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + EPS)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    # match: (xf * rstd).to(bf16) * w  (bf16 multiply, exact in fp32 then rounded)
    normed = (xf * rstd).to(tl.bfloat16)
    y_bf = ((normed.to(tl.float32)) * (w.to(tl.float32))).to(tl.bfloat16)

    # Softmax in fp32 accumulation (matches PyTorch acc_type behavior)
    yf = y_bf.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    row_max = tl.max(yf, axis=0)
    e = tl.math.exp(yf - row_max)
    e = tl.where(mask, e, 0.0)
    denom = tl.sum(e, axis=0)
    out = (e / denom).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 GEMM (tensor cores)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(N)
        _rmsnorm_softmax_kernel[(Mrows,)](
            x, self.rms1_w, y,
            N,
            x.stride(0), y.stride(0),
            EPS=1e-6,
            BLOCK_N=BLOCK_N,
            num_warps=16,
        )
        return y
