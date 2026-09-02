import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 218
M, D, DT = 2048, 1025, torch.bfloat16


@triton.jit
def _fused_rms_gelu_kernel(
    X, W, Y,
    stride_xm,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)  # bf16
    xf = x.to(tl.float32)

    # RMS norm (computed in fp32, result rounded to bf16 like reference)
    ms = tl.sum(xf * xf, axis=0) / N
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (xf * inv).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)  # bf16

    # x * w  (fp32 math, bf16 rounding, matching PyTorch elementwise semantics)
    v = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    # * 1.1772
    v = (v.to(tl.float32) * 1.1772).to(tl.bfloat16)
    # * 1.3155
    v = (v.to(tl.float32) * 1.3155).to(tl.bfloat16)

    # exact gelu (erf) in fp32, round to bf16
    vf = v.to(tl.float32)
    g = 0.5 * vf * (1.0 + tl.math.erf(vf * 0.7071067811865476))
    out = g.to(tl.bfloat16)

    tl.store(Y + row * stride_xm + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1025, 512, generator=g) / math.sqrt(1025)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        x = x.contiguous()
        m, n = x.shape
        y = torch.empty_like(x)
        BLOCK_N = triton.next_power_of_2(n)
        _fused_rms_gelu_kernel[(m,)](
            x, self.rms1_w, y,
            x.stride(0),
            N=n,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
        return y
