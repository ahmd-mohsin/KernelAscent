import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 518
M, D, DT = 2048, 2049, torch.bfloat16


@triton.jit
def _fused_rms_rms_bias_scale_ln(
    x_ptr, w0_ptr, w1_ptr, b2_ptr, g_ptr, b_ptr, out_ptr,
    N, x_stride, out_stride,
    RMS_EPS: tl.constexpr, LN_EPS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # ---- Load row in fp32 ----
    x = tl.load(x_ptr + row * x_stride + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 ----
    ms0 = tl.sum(x * x, axis=0) / N
    y = x * tl.math.rsqrt(ms0 + RMS_EPS)
    y = y.to(tl.bfloat16).to(tl.float32)                      # cast to bf16 like reference
    w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y * w0).to(tl.bfloat16).to(tl.float32)               # bf16 elementwise mul

    # ---- RMSNorm 1 ----
    ms1 = tl.sum(y * y, axis=0) / N
    z = y * tl.math.rsqrt(ms1 + RMS_EPS)
    z = z.to(tl.bfloat16).to(tl.float32)
    w1 = tl.load(w1_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z * w1).to(tl.bfloat16).to(tl.float32)

    # ---- + bias, * scalar (bf16 rounding after each op, matching eager) ----
    b2 = tl.load(b2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z + b2).to(tl.bfloat16).to(tl.float32)
    z = (z * SCALE).to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation) ----
    mean = tl.sum(tl.where(mask, z, 0.0), axis=0) / N
    zc = tl.where(mask, z - mean, 0.0)
    var = tl.sum(zc * zc, axis=0) / N
    inv = tl.math.rsqrt(var + LN_EPS)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = zc * inv * g + b

    tl.store(out_ptr + row * out_stride + offs, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.b2 = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_rms_rms_bias_scale_ln[(rows,)](
            x2, self.rms0_w, self.rms1_w, self.b2, self.ln4_g, self.ln4_b, out,
            N, x2.stride(0), out.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5, SCALE=1.1767,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)

    def _forward_ref(self, x):
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
        x = x + self.b2
        x = x * 1.1767
        x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
        return x
