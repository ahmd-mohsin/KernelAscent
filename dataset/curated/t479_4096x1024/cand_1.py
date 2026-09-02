import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 479
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_scale_ln_bias(
    X, G, B, B3, Y,
    stride_xm, stride_ym,
    N, eps, scale,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    # replicate fp16 elementwise scale (compute in fp32, round to fp16)
    x = (x.to(tl.float32) * scale).to(tl.float16)
    xf = x.to(tl.float32)

    mean = tl.sum(xf, axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xf - mean) * rstd * g + b
    y16 = y.to(tl.float16)

    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    out = y16 + b3
    tl.store(Y + row * stride_ym + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 1024, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0  # cuBLAS fp16 GEMM
        Mrows, N = h.shape
        y = torch.empty_like(h)
        BLOCK_N = triton.next_power_of_2(N)
        _fused_scale_ln_bias[(Mrows,)](
            h, self.ln2_g, self.ln2_b, self.b3, y,
            h.stride(0), y.stride(0),
            N, 1e-5, 1.3078,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )
        return y
