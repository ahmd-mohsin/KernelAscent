import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 213
M, D, DT = 1024, 4097, torch.float16


@triton.jit
def _rms_gelu_kernel(X, W, Y, N, stride_x, stride_y, EPS: tl.constexpr,
                     SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    # RMS norm in fp32
    ms = tl.sum(x * x, axis=0) / N
    rstd = 1.0 / tl.sqrt(ms + EPS)
    xn = (x * rstd).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    h = (xn * w).to(tl.float16)
    h = (h * SCALE).to(tl.float16)
    # exact GELU in fp32 (matches F.gelu on fp16 input up to fp32 math)
    hf = h.to(tl.float32)
    g = hf * 0.5 * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    tl.store(Y + row * stride_y + cols, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 2048, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0
        Mr, N = h.shape
        y = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _rms_gelu_kernel[(Mr,)](
            h, self.rms1_w, y, N,
            h.stride(0), y.stride(0),
            EPS=1e-6, SCALE=1.2985, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
