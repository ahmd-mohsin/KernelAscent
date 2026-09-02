import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 387
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _fused_bias_scale_rms_kernel(
    Y, B, W, OUT,
    N, stride_y, stride_o,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * stride_y + cols, mask=mask, other=0.0)  # fp16
    b = tl.load(B + cols, mask=mask, other=0.0)                   # fp16

    # x = x + b1  (opmath fp32, round to fp16)
    t = (y.to(tl.float32) + b.to(tl.float32)).to(tl.float16)
    # x = x * 1.0435 (opmath fp32, round to fp16)
    s = (t.to(tl.float32) * SCALE).to(tl.float16)

    # RMSNorm in fp32
    xf = s.to(tl.float32)
    xf_m = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf_m * xf_m, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)
    h = (xf * inv).to(tl.float16)

    # * rms3_w (opmath fp32, round to fp16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    out = (h.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # cuBLAS fp16 GEMM (same as reference)
        m, n = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(n)
        _fused_bias_scale_rms_kernel[(m,)](
            y, self.b1, self.rms3_w, out,
            n, y.stride(0), out.stride(0),
            SCALE=1.0435, EPS=1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
