import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 49
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_kernel(x_ptr, b_ptr, w_ptr, out_ptr, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * D + cols, mask=mask, other=0.0)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0)

    # x = x + b0  (fp16 add, matching PyTorch half add which uses fp32 opmath then rounds)
    xf = x.to(tl.float32) + b.to(tl.float32)
    x16 = xf.to(tl.float16)

    # RMSNorm in fp32
    xf = x16.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    n16 = (xf * inv).to(tl.float16)

    # multiply by weight (fp16 tensor mul: fp32 opmath, round to fp16)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    t16 = (n16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # scalar multiply (fp32 opmath, round to fp16)
    s16 = (t16.to(tl.float32) * 1.2274).to(tl.float16)

    # exact GELU (fp32 opmath like PyTorch on half), round to fp16
    g = s16.to(tl.float32)
    y = 0.5 * g * (1.0 + tl.math.erf(g * 0.7071067811865476))
    tl.store(out_ptr + row * D + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.b0, self.rms1_w, out, d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
