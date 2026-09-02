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
    X, W0, W2, Out,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 (float accumulate, fp16 scale-mul like reference) ----
    ms = tl.sum(x * x, axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    xh = (x * inv).to(tl.float16)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0)
    xh = xh * w0

    # ---- GELU (exact erf, float opmath like PyTorch half gelu) ----
    xf = xh.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))
    xh = xf.to(tl.float16)

    # ---- RMSNorm 2 ----
    xf = xh.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    inv = tl.math.rsqrt(ms + 1e-6)
    xh = (xf * inv).to(tl.float16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    xh = xh * w2

    # ---- Softmax (float accumulate like PyTorch half softmax) ----
    xf = xh.to(tl.float32)
    xf = tl.where(mask, xf, float('-inf'))
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    xh = (e / s).to(tl.float16)

    # ---- GELU ----
    xf = xh.to(tl.float32)
    xf = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))

    tl.store(Out + row * D + offs, xf.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            return self._ref_forward(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4

        _fused_rms_gelu_rms_softmax_gelu[(n_rows,)](
            x2, self.rms0_w, self.rms2_w, out,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

    def _ref_forward(self, x):
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = F.gelu(x)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        x = torch.softmax(x, dim=-1)
        x = F.gelu(x)
        return x
