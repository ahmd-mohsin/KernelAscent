import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 438
M, D, DT = 1024, 512, torch.bfloat16


@triton.jit
def _fused_rms_kernel(X, W, B, Y,
                      stride_xm,
                      N: tl.constexpr,
                      BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # RMS statistic in fp32 (matches _xf.pow(2).mean(-1) + rsqrt)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + 1e-6)

    # (xf * rsqrt).to(bf16)
    t = (x * rstd).to(tl.bfloat16)

    w = tl.load(W + cols, mask=mask, other=0.0)
    b = tl.load(B + cols, mask=mask, other=0.0)

    # bf16 elementwise ops with rounding after each step (matches PyTorch)
    t = (t.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)
    t = (t.to(tl.float32) + b.to(tl.float32)).to(tl.bfloat16)
    t = (t.to(tl.float32) * 1.1065).to(tl.bfloat16)
    t = (t.to(tl.float32) * 1.1693).to(tl.bfloat16)
    t = tl.maximum(t, 0.0)

    tl.store(Y + row * stride_xm + cols, t, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 512, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0  # tensor-core bf16 GEMM
        y = y.contiguous()
        m, n = y.shape
        out = torch.empty_like(y)
        _fused_rms_kernel[(m,)](
            y, self.rms1_w, self.b2, out,
            y.stride(0),
            N=n, BLOCK=triton.next_power_of_2(n),
            num_warps=4,
        )
        return out
