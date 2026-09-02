import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 271
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_kernel(X, B1, W, Out,
                  stride_xm, stride_om,
                  N, eps,
                  BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu (exact, erf-based), computed in fp32, rounded to fp16 like PyTorch half ops
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # add bias
    b = tl.load(B1 + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b)
    x = x.to(tl.float16).to(tl.float32)

    # gelu again
    x = x * 0.5 * (1.0 + tl.math.erf(x * INV_SQRT2))
    x = x.to(tl.float16).to(tl.float32)

    # rmsnorm (fp32 accumulate, matching reference)
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + eps)
    xn = (x * rstd).to(tl.float16).to(tl.float32)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.float16)

    tl.store(Out + row * stride_om + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, n = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(n)
        _fused_kernel[(m,)](
            x, self.b1, self.rms3_w, out,
            x.stride(0), out.stride(0),
            n, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
