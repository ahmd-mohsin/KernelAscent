import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 197
M, D, DT = 8192, 513, torch.float16


@triton.jit
def _gelu_rms_kernel(
    X_ptr, W_ptr, Out_ptr,
    stride_xm,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X_ptr + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # exact (erf-based) GELU in fp32, then round to fp16 (matches F.gelu on half)
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    gh = g.to(tl.float16)

    # RMSNorm: cast to fp32, mean of squares, rsqrt
    gf = gh.to(tl.float32)
    ms = tl.sum(gf * gf, axis=0) / N
    inv = tl.math.rsqrt(ms + eps)
    normed = (gf * inv).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0)
    # half * half elementwise in PyTorch uses fp32 opmath
    out = (normed.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    tl.store(Out_ptr + row * stride_xm + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(513, 4096, generator=g) / math.sqrt(513)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        m, n = y.shape
        out = torch.empty_like(y)
        _gelu_rms_kernel[(m,)](
            y, self.rms2_w, out,
            y.stride(0),
            n, 1e-6,
            BLOCK_N=4096,
            num_warps=8,
        )
        return out
