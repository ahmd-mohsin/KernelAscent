import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 935
M, D, DT = 4096, 4097, torch.float16


@triton.jit
def _fused_gelu_rms_gelu2_kernel(
    Y_ptr, W_ptr, OUT_ptr,
    N, stride_ym, stride_om,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y_ptr + row * stride_ym + cols, mask=mask, other=0.0).to(tl.float32)

    # gelu (exact, erf), computed in fp32, rounded to fp16 (matches PyTorch half gelu)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    g = y * 0.5 * (1.0 + tl.math.erf(y * INV_SQRT2))
    g16 = g.to(tl.float16)
    gf = g16.to(tl.float32)

    # RMSNorm in fp32
    ms = tl.sum(tl.where(mask, gf * gf, 0.0), axis=0) / N
    rs = tl.math.rsqrt(ms + EPS)
    t = (gf * rs).to(tl.float16)

    w = tl.load(W_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    z = (t.to(tl.float32) * w).to(tl.float16)

    # gelu twice, each computed in fp32 with fp16 rounding between
    zf = z.to(tl.float32)
    z2 = (zf * 0.5 * (1.0 + tl.math.erf(zf * INV_SQRT2))).to(tl.float16)
    z2f = z2.to(tl.float32)
    z3 = (z2f * 0.5 * (1.0 + tl.math.erf(z2f * INV_SQRT2))).to(tl.float16)

    tl.store(OUT_ptr + row * stride_om + cols, z3, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4097, 4096, generator=g) / math.sqrt(4097)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = torch.matmul(x, self.W0)
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        BLOCK = triton.next_power_of_2(N)
        _fused_gelu_rms_gelu2_kernel[(Mrows,)](
            y, self.rms2_w, out,
            N, y.stride(0), out.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
