import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 230
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_kernel(X, W0, W2, Y, D_dim, stride_x, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_dim

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 0 (fp32 math, fp16 store semantics)
    ms = tl.sum(x * x, axis=0) / D_dim
    inv = tl.math.rsqrt(ms + 1e-6)
    y16 = (x * inv).to(tl.float16)
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    y16 = y16 * w0  # fp16 multiply, fp16 result

    # GELU (exact, erf) computed in fp32, stored as fp16
    yf = y16.to(tl.float32)
    g = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    g16 = g.to(tl.float16)

    # RMSNorm 2
    gf = g16.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / D_dim
    inv2 = tl.math.rsqrt(ms2 + 1e-6)
    z16 = (gf * inv2).to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    z16 = z16 * w2

    # Softmax (fp32 accumulation, fp16 output)
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, float('-inf'))
    zmax = tl.max(zf, axis=0)
    e = tl.exp(zf - zmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm16 = (e / s).to(tl.float16)

    # GELU again
    sf = sm16.to(tl.float32)
    out = sf * 0.5 * (1.0 + tl.math.erf(sf * 0.7071067811865476))
    out16 = out.to(tl.float16)

    tl.store(Y + row * stride_y + cols, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = F.gelu(x)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = torch.softmax(x, dim=-1)
            x = F.gelu(x)
            return x

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        rows, dim = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(dim)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.rms0_w, self.rms2_w, y,
            dim, x2.stride(0), y.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
