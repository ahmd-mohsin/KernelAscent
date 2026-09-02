import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 682
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_kernel(
    X, W0, W4, OUT,
    N, stride_x, stride_o,
    eps, s1, s2,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 1
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    a = (x * r).to(tl.float16).to(tl.float32)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0).to(tl.float32)
    a = (a * w0).to(tl.float16).to(tl.float32)

    # exact GELU (erf), computed in fp32 (matches PyTorch opmath), round to fp16
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = a * 0.5 * (1.0 + tl.math.erf(a * INV_SQRT2))
    g = g.to(tl.float16).to(tl.float32)

    # scalar multiplies with fp16 rounding between (matches two separate ops)
    b = (g * s1).to(tl.float16).to(tl.float32)
    c = (b * s2).to(tl.float16).to(tl.float32)

    # RMSNorm 2
    c_masked = tl.where(mask, c, 0.0)
    ms2 = tl.sum(c_masked * c_masked, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + eps)
    d = (c * r2).to(tl.float16).to(tl.float32)
    w4 = tl.load(W4 + cols, mask=mask, other=0.0).to(tl.float32)
    out = (d * w4).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = F.gelu(x)
            x = x * 1.0992
            x = x * 1.3755
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        n = orig_shape[-1]
        x2d = x.contiguous().view(-1, n)
        m = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_kernel[(m,)](
            x2d, self.rms0_w, self.rms4_w, out,
            n, x2d.stride(0), out.stride(0),
            1e-6, 1.0992, 1.3755,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
