import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 425
M, D, DT = 4096, 513, torch.float16


@triton.jit
def _fused_rms_kernel(
    X, W, B, Y,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)  # fp16

    # x = x * 1.3008  (PyTorch computes half ops in fp32 opmath, rounds to fp16)
    x_scaled_f32 = x.to(tl.float32) * 1.3008
    x_scaled = x_scaled_f32.to(tl.float16)

    # _xf = x.float(); rms
    xf = x_scaled.to(tl.float32)
    xf_masked = tl.where(mask, xf, 0.0)
    mean_sq = tl.sum(xf_masked * xf_masked, axis=0) / N
    inv = tl.math.rsqrt(mean_sq + 1e-6)

    normed = (xf * inv).to(tl.float16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # fp16 * fp16 with fp32 opmath, round to fp16
    t = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.float16)
    out = (t.to(tl.float32) + b.to(tl.float32)).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_rms_kernel[(Mrows,)](
            x, self.rms1_w, self.b2, y,
            N, x.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y
