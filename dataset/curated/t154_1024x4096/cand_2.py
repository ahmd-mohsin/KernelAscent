import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 154
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _fused_kernel(
    X, LN_G, LN_B, B3, RMS_W, OUT,
    stride_x, stride_o,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    # relu (in fp16, same result as fp32 for relu)
    x = tl.maximum(x, 0.0)
    # gelu exact (erf), computed in fp32 then cast to fp16 (matches PyTorch opmath)
    xf = x.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = xf * 0.5 * (1.0 + tl.math.erf(xf * INV_SQRT2))
    g16 = g.to(tl.float16)

    # layernorm in fp32 accumulation (matches PyTorch half layernorm)
    gf = g16.to(tl.float32)
    gf = tl.where(mask, gf, 0.0)
    mean = tl.sum(gf, axis=0) / N
    diff = tl.where(mask, gf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    w = tl.load(LN_G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN_B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (gf - mean) * rstd * w + b
    y16 = y.to(tl.float16)

    # add b3 in fp16
    b3 = tl.load(B3 + cols, mask=mask, other=0.0)
    z16 = y16 + b3

    # rmsnorm: fp32 math, cast to fp16, then multiply by weight in fp16
    zf = z16.to(tl.float32)
    zf_masked = tl.where(mask, zf, 0.0)
    ms = tl.sum(zf_masked * zf_masked, axis=0) / N
    rrms = tl.math.rsqrt(ms + 1e-6)
    out16 = (zf * rrms).to(tl.float16)
    rw = tl.load(RMS_W + cols, mask=mask, other=0.0)
    out16 = out16 * rw

    tl.store(OUT + row * stride_o + cols, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b3 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.relu(x)
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = x + self.b3
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(rows,)](
            x2, self.ln2_g, self.ln2_b, self.b3, self.rms4_w, out,
            x2.stride(0), out.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
