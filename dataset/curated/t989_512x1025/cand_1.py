import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 989
M, D, DT = 512, 1025, torch.bfloat16


@triton.jit
def _fused_kernel(
    x_ptr, b1_ptr, ln_g_ptr, ln_b_ptr, rms_w_ptr, out_ptr,
    D, x_stride, out_stride,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(x_ptr + row * x_stride + cols, mask=mask, other=0.0).to(tl.float32)

    # x = x * 1.4347 (bf16 op: fp32 math, round to bf16)
    x = (x * 1.4347).to(tl.bfloat16).to(tl.float32)

    # x = x + b1
    b1 = tl.load(b1_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = (x + b1).to(tl.bfloat16).to(tl.float32)

    # exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = g.to(tl.bfloat16).to(tl.float32)

    # LayerNorm (eps=1e-5), fp32 math, output rounded to bf16
    xm = tl.where(mask, x, 0.0)
    mean = tl.sum(xm, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    gamma = tl.load(ln_g_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    beta = tl.load(ln_b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((xc * rstd) * gamma + beta).to(tl.bfloat16).to(tl.float32)

    # RMSNorm (explicit fp32, eps=1e-6), cast to bf16, then * w (bf16 op)
    ym = tl.where(mask, y, 0.0)
    ms = tl.sum(ym * ym, axis=0) / D
    r = (y * tl.math.rsqrt(ms + 1e-6)).to(tl.bfloat16).to(tl.float32)
    w = tl.load(rms_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = (r * w).to(tl.bfloat16)

    tl.store(out_ptr + row * out_stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b1 = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            y = x * 1.4347
            y = y + self.b1
            y = F.gelu(y)
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            _yf = y.float()
            y = (_yf * torch.rsqrt(_yf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms4_w
            return y

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            x2, self.b1, self.ln3_g, self.ln3_b, self.rms4_w, out,
            d, x2.stride(0), out.stride(0),
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
