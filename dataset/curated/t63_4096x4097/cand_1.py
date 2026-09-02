import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 63
M, D, DT = 4096, 4097, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b0_ptr, w2_ptr, w3_ptr, b4_ptr, out_ptr,
    D_dim, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_dim

    base = row * stride_row

    # x + b0 (fp32 compute, round to bf16 like PyTorch opmath)
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf), fp32 compute then round to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm 1
    ms1 = tl.sum(tl.where(mask, g * g, 0.0), axis=0) / D_dim
    r1 = tl.math.rsqrt(ms1 + 1e-6)
    y = (g * r1).to(tl.bfloat16)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0)
    y = (y.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # RMSNorm 2
    yf = y.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / D_dim
    r2 = tl.math.rsqrt(ms2 + 1e-6)
    z = (yf * r2).to(tl.bfloat16)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0)
    z = (z.to(tl.float32) * w3.to(tl.float32)).to(tl.bfloat16)

    # + b4
    b4 = tl.load(b4_ptr + offs, mask=mask, other=0.0)
    out = (z.to(tl.float32) + b4.to(tl.float32)).to(tl.bfloat16)

    tl.store(out_ptr + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_kernel[(rows,)](
            x2, self.b0, self.rms2_w, self.rms3_w, self.b4, out,
            d, x2.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
