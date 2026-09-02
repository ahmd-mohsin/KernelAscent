import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 535
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _rms_gelu2_kernel(X, W, Y, N, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)

    # normalize in fp32, round to bf16 (matches .to(x.dtype))
    xn = (x * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    # bf16 multiply: compute in fp32, round to bf16
    h = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # gelu (exact, erf) computed in fp32, rounded to bf16
    hf = h.to(tl.float32)
    g1 = (0.5 * hf * (1.0 + tl.math.erf(hf * 0.7071067811865476))).to(tl.bfloat16)

    g1f = g1.to(tl.float32)
    g2 = (0.5 * g1f * (1.0 + tl.math.erf(g1f * 0.7071067811865476))).to(tl.bfloat16)

    tl.store(Y + row * stride_y + cols, g2, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0  # cuBLAS bf16 matmul
        Mrows, N = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _rms_gelu2_kernel[(Mrows,)](
            x, self.rms1_w, y, N,
            x.stride(0), y.stride(0),
            1e-6, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
