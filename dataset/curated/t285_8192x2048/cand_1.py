import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 285
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _bias_rmsnorm_kernel(
    X, B, W, Y,
    stride_x, stride_y,
    N, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)

    # x + b1 : PyTorch computes bf16+bf16 in fp32 opmath then rounds to bf16
    xb = (x + b).to(X.dtype.element_ty)
    xf = xb.to(tl.float32)

    # RMS over the row in fp32 (matches _xf.pow(2).mean(-1) + eps, rsqrt)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + eps)

    # (_xf * rsqrt(...)).to(bf16)  -> then bf16 * bf16 weight (fp32 opmath, one round)
    y = (xf * r).to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(Y.dtype.element_ty)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        y = torch.matmul(x, self.W0)
        y = y.contiguous()

        Mrows, N = y.shape
        BLOCK = triton.next_power_of_2(N)

        out = torch.empty_like(y)
        _bias_rmsnorm_kernel[(Mrows,)](
            y, self.b1, self.rms2_w, out,
            y.stride(0), out.stride(0),
            N, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
