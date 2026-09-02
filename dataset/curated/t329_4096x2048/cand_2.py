import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 329
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _rms_gelu_scale_kernel(
    X, W, Y,
    N, stride_x, stride_y,
    eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMS norm in fp32
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + eps)
    xn = (xf * inv).to(tl.float16)  # round to fp16 like .to(x.dtype)

    # multiply by weight (fp16 elementwise -> rounds to fp16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    xw = (xn * w).to(tl.float16)

    # exact (erf-based) GELU with fp32 internal math, cast back to fp16
    xg32 = xw.to(tl.float32)
    g = 0.5 * xg32 * (1.0 + tl.math.erf(xg32 * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # scale by 1.203 (fp16 arithmetic)
    out = (g16.to(tl.float32) * scale)
    out16 = out.to(tl.float16)

    tl.store(Y + row * stride_y + cols, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.W4 = nn.Parameter((torch.randn(512, 1024, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS)
        h = x @ self.W0  # (M, 512) fp16

        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _rms_gelu_scale_kernel[(Mrows,)](
            h, self.rms1_w, y,
            N, h.stride(0), y.stride(0),
            1e-6, 1.203,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )

        # GEMM 2 (cuBLAS) + in-place ReLU
        out = y @ self.W4
        return torch.relu_(out)
