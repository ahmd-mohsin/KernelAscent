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
    N, stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(x_ptr + row * stride_row + offs, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(b0_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # x = x + b0  (bf16 arithmetic: fp32 compute, round to bf16)
    x = (x + b0).to(tl.bfloat16).to(tl.float32)

    # exact GELU (erf-based), computed in fp32, rounded to bf16
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)

    # RMSNorm 1 (stats in fp32)
    gm = tl.where(mask, g, 0.0)
    ms1 = tl.sum(gm * gm, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    y = (g * r1).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y * w2).to(tl.bfloat16).to(tl.float32)

    # RMSNorm 2
    ym = tl.where(mask, y, 0.0)
    ms2 = tl.sum(ym * ym, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    z = (y * r2).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z * w3).to(tl.bfloat16).to(tl.float32)

    # z = z + b4
    b4 = tl.load(b4_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z + b4).to(tl.bfloat16)

    tl.store(out_ptr + row * stride_row + offs, z, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.bfloat16:
            # fallback (reference path)
            x = x + self.b0
            x = F.gelu(x)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            x = x + self.b4
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 4096 else 4

        _fused_kernel[(rows,)](
            x2, self.b0, self.rms2_w, self.rms3_w, self.b4, out,
            N, x2.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
