import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 588
M, D, DT = 4096, 2049, torch.bfloat16


@triton.jit
def _double_rmsnorm_kernel(
    x_ptr, w0_ptr, w1_ptr, out_ptr,
    D: tl.constexpr, eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- first RMSNorm ----
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    ms0 = tl.sum(x * x, axis=0) / D
    rr0 = tl.math.rsqrt(ms0 + eps)
    y0 = (x * rr0).to(tl.bfloat16)          # round to bf16 (matches .to(x.dtype))
    w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0)
    y0 = y0 * w0                            # bf16 multiply (exact in fp32, single rounding)

    # ---- second RMSNorm ----
    x1 = y0.to(tl.float32)
    ms1 = tl.sum(x1 * x1, axis=0) / D
    rr1 = tl.math.rsqrt(ms1 + eps)
    y1 = (x1 * rr1).to(tl.bfloat16)
    w1 = tl.load(w1_ptr + offs, mask=mask, other=0.0)
    y1 = y1 * w1

    tl.store(out_ptr + base + offs, y1, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not (x.is_cuda and x.dtype == torch.bfloat16):
            # fallback: reference path
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        m = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(d)
        _double_rmsnorm_kernel[(m,)](
            x2d, self.rms0_w, self.rms1_w, out,
            d, 1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out.view(orig_shape)
