import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 422
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _relu_scale_rms_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # fp16
    # relu in fp16 (bit-exact regardless of dtype)
    x = tl.maximum(x, 0.0)
    # x * 1.0221: PyTorch computes half*scalar in fp32 then rounds to fp16
    xf = x.to(tl.float32) * 1.0221
    x16 = xf.to(tl.float16)          # rounded fp16 value (matches reference tensor x)
    xf = x16.to(tl.float32)          # _xf = x.float()

    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    inv = tl.math.rsqrt(ms + 1e-6)

    norm16 = (xf * inv).to(tl.float16)               # (...).to(x.dtype)
    w = tl.load(W + cols, mask=mask, other=0.0)      # fp16
    # fp16 * fp16 elementwise: computed in fp32, rounded to fp16
    out = (norm16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS fp16 matmul (tensor cores)
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _relu_scale_rms_kernel[(Mrows,)](
            x, self.rms3_w, y,
            x.stride(0), y.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return y
