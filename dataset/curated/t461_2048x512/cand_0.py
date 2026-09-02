import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 461
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_scale_rmsnorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, stride_x, stride_y,
    SCALE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32) * SCALE

    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + EPS)

    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    b = tl.load(B_ptr + cols, mask=mask, other=0.0)

    y = xn * w + b  # bf16 ops to match reference semantics
    tl.store(Y_ptr + row * stride_y + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS bf16 matmul (tensor cores)
        Mrows, N = h.shape
        out = torch.empty_like(h)
        _fused_scale_rmsnorm_kernel[(Mrows,)](
            h, self.rms2_w, self.b3, out,
            N, h.stride(0), out.stride(0),
            SCALE=1.0961, EPS=1e-6,
            BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )
        return out
