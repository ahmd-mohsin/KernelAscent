import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 400
M, D, DT = 2048, 2048, torch.float16


@triton.jit
def _fused_rms_softmax_rms_kernel(
    x_ptr, w0_ptr, w2_ptr, out_ptr,
    D: tl.constexpr,
    EPS: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    base = row * D

    # ---- RMSNorm 1 (compute in fp32, round to fp16, then * weight in fp32 opmath) ----
    x = tl.load(x_ptr + base + offs)
    xf = x.to(tl.float32)
    inv_rms = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / D + EPS)
    h16 = (xf * inv_rms).to(tl.float16)
    w0 = tl.load(w0_ptr + offs)
    h16 = (h16.to(tl.float32) * w0.to(tl.float32)).to(tl.float16)

    # ---- Softmax (fp32 accumulation, fp16 output; matches PyTorch half softmax) ----
    sf = h16.to(tl.float32)
    m = tl.max(sf, axis=0)
    e = tl.exp(sf - m)
    s = tl.sum(e, axis=0)
    sm16 = (e / s).to(tl.float16)

    # ---- RMSNorm 2 ----
    sf2 = sm16.to(tl.float32)
    inv_rms2 = tl.math.rsqrt(tl.sum(sf2 * sf2, axis=0) / D + EPS)
    y16 = (sf2 * inv_rms2).to(tl.float16)
    w2 = tl.load(w2_ptr + offs)
    y16 = (y16.to(tl.float32) * w2.to(tl.float32)).to(tl.float16)

    # ---- Two scalar multiplies (fp32 opmath, rounded to fp16 each step, like PyTorch) ----
    y16 = (y16.to(tl.float32) * S1).to(tl.float16)
    y16 = (y16.to(tl.float32) * S2).to(tl.float16)

    tl.store(out_ptr + base + offs, y16)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float16 and x.shape[-1] == 2048:
            orig_shape = x.shape
            x2d = x.contiguous().view(-1, 2048)
            rows = x2d.shape[0]
            out = torch.empty_like(x2d)
            _fused_rms_softmax_rms_kernel[(rows,)](
                x2d, self.rms0_w, self.rms2_w, out,
                D=2048, EPS=1e-6, S1=1.3308, S2=1.3952,
                num_warps=8,
            )
            return out.view(orig_shape)

        # Fallback (reference path)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = torch.softmax(x, dim=-1)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        x = x * 1.3308
        x = x * 1.3952
        return x
