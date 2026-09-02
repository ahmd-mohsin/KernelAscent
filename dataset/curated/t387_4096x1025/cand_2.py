import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 387
M, D, DT = 4096, 1025, torch.float16


@triton.jit
def _fused_bias_scale_rmsnorm(
    X_ptr, B_ptr, W_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x + b1  (fp16 op: compute in fp32, round to fp16)
    x = (x + b).to(tl.float16).to(tl.float32)
    # x = x * 1.0435 (fp16 op: compute in fp32, round to fp16)
    x = (x * SCALE).to(tl.float16).to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(x * x, axis=0) / N
    r = tl.math.rsqrt(ms + EPS)

    # (xf * rsqrt).to(fp16)
    y = (x * r).to(tl.float16).to(tl.float32)

    # * rms3_w (fp16 op: compute in fp32, round to fp16)
    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (y * w).to(tl.float16)

    tl.store(Y_ptr + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 2048, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # cuBLAS fp16 matmul (tensor cores on A100)
        h = x @ self.W0
        h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_bias_scale_rmsnorm[(Mrows,)](
            h, self.b1, self.rms3_w, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.0435,
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
