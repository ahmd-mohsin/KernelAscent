import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 662
M, D, DT = 512, 513, torch.bfloat16


@triton.jit
def _fused_rms_bias_rms_gelu(
    x_ptr, w0_ptr, b1_ptr, w2_ptr, out_ptr,
    D, stride_x, stride_o, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- load row in fp32 ----
    x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 ----
    ms = tl.sum(x * x, axis=0) / D
    rr = tl.math.rsqrt(ms + eps)
    # cast to bf16 (matches .to(x.dtype)), back to fp32 for exact bf16-emulated arithmetic
    xn = (x * rr).to(tl.bfloat16).to(tl.float32)

    w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # bf16 * bf16 -> bf16 is exact in fp32 then rounded once
    x1 = (xn * w0).to(tl.bfloat16).to(tl.float32)

    # ---- bias add (bf16 add emulated exactly in fp32 + single round) ----
    b1 = tl.load(b1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = (x1 + b1).to(tl.bfloat16).to(tl.float32)
    x2 = tl.where(mask, x2, 0.0)

    # ---- RMSNorm 2 ----
    ms2 = tl.sum(x2 * x2, axis=0) / D
    rr2 = tl.math.rsqrt(ms2 + eps)
    x3 = (x2 * rr2).to(tl.bfloat16).to(tl.float32)

    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x4 = (x3 * w2).to(tl.bfloat16).to(tl.float32)

    # ---- GELU (erf variant, computed in fp32 as PyTorch does for bf16) ----
    g = x4 * 0.5 * (1.0 + tl.math.erf(x4 * 0.7071067811865476))

    tl.store(out_ptr + row * stride_o + offs, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x + self.b1
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return F.gelu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        _fused_rms_bias_rms_gelu[(m,)](
            x2d, self.rms0_w, self.b1, self.rms2_w, out,
            d, x2d.stride(0), out.stride(0), 1e-6,
            BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return out.view(orig_shape)
