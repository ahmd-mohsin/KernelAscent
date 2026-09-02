import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 230
M, D, DT = 1024, 2048, torch.float16


@triton.jit
def _fused_rms_gelu_rms_softmax_gelu(
    X, W0, W2, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    SQRT1_2: tl.constexpr = 0.7071067811865476

    # ---- RMSNorm 0 (fp32 math, cast to fp16 like reference) ----
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.rsqrt(tl.sum(x * x, axis=0) / D + 1e-6)
    h = (x * r).to(tl.float16)

    w0 = tl.load(W0 + offs, mask=mask, other=0.0)
    # half*half elementwise -> computed in fp32 (opmath), stored fp16
    h = (h.to(tl.float32) * w0.to(tl.float32)).to(tl.float16)

    # ---- GELU (exact erf, fp32 opmath, cast fp16) ----
    hf = h.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * SQRT1_2))
    h = g.to(tl.float16)

    # ---- RMSNorm 2 ----
    hf = h.to(tl.float32)
    r = tl.rsqrt(tl.sum(hf * hf, axis=0) / D + 1e-6)
    h = (hf * r).to(tl.float16)

    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    h = (h.to(tl.float32) * w2.to(tl.float32)).to(tl.float16)

    # ---- Softmax (fp32 accumulation, cast fp16) ----
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, float('-inf'))
    m = tl.max(hf, axis=0)
    e = tl.exp(hf - m)
    e = tl.where(mask, e, 0.0)
    s = e / tl.sum(e, axis=0)
    h = s.to(tl.float16)

    # ---- GELU ----
    hf = h.to(tl.float32)
    g = 0.5 * hf * (1.0 + tl.math.erf(hf * SQRT1_2))

    tl.store(Y + row * D + offs, g.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if x.is_cuda and x.dtype == torch.float16 and x.shape[-1] == self.rms0_w.numel():
            orig_shape = x.shape
            d = orig_shape[-1]
            x2 = x.contiguous().view(-1, d)
            m = x2.shape[0]
            y = torch.empty_like(x2)
            BLOCK = triton.next_power_of_2(d)
            _fused_rms_gelu_rms_softmax_gelu[(m,)](
                x2, self.rms0_w, self.rms2_w, y,
                D=d, BLOCK=BLOCK,
                num_warps=8,
            )
            return y.view(orig_shape)

        # Fallback (numerically identical reference path)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = F.gelu(x)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        x = torch.softmax(x, dim=-1)
        x = F.gelu(x)
        return x
