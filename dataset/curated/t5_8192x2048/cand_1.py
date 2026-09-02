import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 5
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, b1_ptr, w_ptr, out_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    base = row * D
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0)
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0)

    # x = x + b0 (fp16 add with float opmath, rounded to fp16)
    t = (x.to(tl.float32) + b0.to(tl.float32)).to(tl.float16)
    # x = x + b1
    t = (t.to(tl.float32) + b1.to(tl.float32)).to(tl.float16)

    xf = t.to(tl.float32)

    # RMS norm in fp32
    ss = tl.sum(xf * xf, axis=0)
    mean = ss / D
    rs = tl.math.rsqrt(mean + 1e-6)

    # (_xf * rsqrt).to(fp16)
    n16 = (xf * rs).to(tl.float16)

    # * rms2_w (fp16 mul, float opmath, round fp16)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    p = (n16.to(tl.float32) * w.to(tl.float32)).to(tl.float16)

    # * 1.287 (float opmath, round fp16)
    s = (p.to(tl.float32) * 1.287).to(tl.float16)

    # gelu exact (float opmath, round fp16)
    sf = s.to(tl.float32)
    g = sf * 0.5 * (1.0 + tl.math.erf(sf * 0.7071067811865476))
    out = g.to(tl.float16)

    tl.store(out_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        m, d = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x, self.b0, self.b1, self.rms2_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return out
