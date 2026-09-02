import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 337
M, D, DT = 1024, 4096, torch.bfloat16


@triton.jit
def _rms_relu_kernel(X, W, Y, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16)  # round to bf16 like .to(x.dtype)
    w = tl.load(W + offs, mask=mask, other=0.0)
    y = xn.to(tl.float32) * w.to(tl.float32)  # bf16 mul (fp32 accum, single rounding)
    y = tl.maximum(y, 0.0)
    tl.store(Y + row * D + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _rms_relu_kernel[(m,)](x, self.rms0_w, y, d, BLOCK=BLOCK, num_warps=8)
        return y
