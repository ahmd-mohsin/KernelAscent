import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 812
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _triple_rmsnorm_kernel(
    x_ptr, w0_ptr, w2_ptr, w3_ptr, out_ptr,
    D: tl.constexpr, EPS: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    xf = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    # --- RMSNorm 0 ---
    ms = tl.sum(xf * xf, axis=0) / D
    rstd = tl.rsqrt(ms + EPS)
    xn = (xf * rstd).to(tl.bfloat16)
    w0 = tl.load(w0_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xn.to(tl.float32) * w0).to(tl.bfloat16)

    # --- scale ---
    x = (x.to(tl.float32) * SCALE).to(tl.bfloat16)

    # --- RMSNorm 2 ---
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    rstd = tl.rsqrt(ms + EPS)
    xn = (xf * rstd).to(tl.bfloat16)
    w2 = tl.load(w2_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xn.to(tl.float32) * w2).to(tl.bfloat16)

    # --- RMSNorm 3 ---
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    rstd = tl.rsqrt(ms + EPS)
    xn = (xf * rstd).to(tl.bfloat16)
    w3 = tl.load(w3_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xn.to(tl.float32) * w3).to(tl.bfloat16)

    tl.store(out_ptr + base + offs, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)
        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        out = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _triple_rmsnorm_kernel[(m,)](
            xc, self.rms0_w, self.rms2_w, self.rms3_w, out,
            D=d, EPS=1e-6, SCALE=1.4041, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

    def _forward_ref(self, x):
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = x * 1.4041
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        return x
