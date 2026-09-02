import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 438
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_rms_bias_scale_relu(
    X, W, B, Y,
    stride_xm,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMS statistic in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)

    # normalize, round to bf16 (matches .to(x.dtype))
    y = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # y = y * w   (bf16 op: fp32 math, bf16 rounding)
    y = (y.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    # y = y + b
    y = (y.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    # y = y * 1.1065
    y = (y.to(tl.float32) * 1.1065).to(tl.bfloat16)
    # y = y * 1.1693
    y = (y.to(tl.float32) * 1.1693).to(tl.bfloat16)
    # relu
    y = tl.maximum(y, 0.0).to(tl.bfloat16)

    tl.store(Y + row * stride_xm + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS matmul (tensor cores)
        x = x.contiguous()
        M_, N_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N_)
        _fused_rms_bias_scale_relu[(M_,)](
            x, self.rms1_w, self.b2, y,
            x.stride(0),
            N=N_,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return y
