import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 682
M, D, DT = 8192, 1025, torch.float16


@triton.jit
def _fused_rms_gelu_rms_kernel(
    X, W0, W4, Y,
    N, stride_x, stride_y,
    EPS: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm 0 (compute in fp32, round to fp16, then multiply by weight)
    ms = tl.sum(x * x, axis=0) / N
    inv = tl.math.rsqrt(ms + EPS)
    xh = (x * inv).to(tl.float16)

    w0 = tl.load(W0 + offs, mask=mask, other=0.0).to(tl.float32)
    h = (xh.to(tl.float32) * w0).to(tl.float16)

    # exact GELU (erf-based), computed in fp32 like PyTorch opmath, rounded to fp16
    hf = h.to(tl.float32)
    g = hf * 0.5 * (1.0 + tl.math.erf(hf * 0.7071067811865476))
    h = g.to(tl.float16)

    # two separate scalar scalings, each with intermediate fp16 rounding
    h = (h.to(tl.float32) * S1).to(tl.float16)
    h = (h.to(tl.float32) * S2).to(tl.float16)

    # RMSNorm 4
    hf = h.to(tl.float32)
    ms2 = tl.sum(hf * hf, axis=0) / N
    inv2 = tl.math.rsqrt(ms2 + EPS)
    yh = (hf * inv2).to(tl.float16)

    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (yh.to(tl.float32) * w4).to(tl.float16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1025, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            return self._forward_ref(x)

        orig_shape = x.shape
        n = orig_shape[-1]
        x2 = x.contiguous().view(-1, n)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(n)
        _fused_rms_gelu_rms_kernel[(rows,)](
            x2, self.rms0_w, self.rms4_w, y,
            n, x2.stride(0), y.stride(0),
            EPS=1e-6, S1=1.0992, S2=1.3755,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = F.gelu(x)
        x = x * 1.0992
        x = x * 1.3755
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
        return x
