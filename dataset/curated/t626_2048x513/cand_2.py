import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 626
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _fused_kernel(
    X, W, G, B, OUT,
    N, stride_x, stride_o,
    RMS_EPS: tl.constexpr,
    LN_EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)

    # ---- x = x * 1.0029 (compute in fp32, round to fp16 like PyTorch) ----
    xf = x.to(tl.float32) * 1.0029
    xh = xf.to(tl.float16)

    # ---- RMSNorm (stats in fp32, cast to fp16, then fp16 multiply by weight) ----
    xf2 = xh.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf2 * xf2, 0.0), axis=0) / N
    rinv = tl.math.rsqrt(ms + RMS_EPS)
    w = tl.load(W + cols, mask=mask, other=0.0)
    yh = (xf2 * rinv).to(tl.float16) * w  # fp16 multiply, matches ref

    # ---- Softmax (fp32 accumulate, fp16 output) ----
    yf = yh.to(tl.float32)
    yf_m = tl.where(mask, yf, float('-inf'))
    mx = tl.max(yf_m, axis=0)
    e = tl.exp(yf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # ---- LayerNorm (fp32 stats, fp32 affine, fp16 output) ----
    pf = sm.to(tl.float32)
    mean = tl.sum(tl.where(mask, pf, 0.0), axis=0) / N
    d = tl.where(mask, pf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + LN_EPS)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = ((pf - mean) * inv * g + b).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xf = (x.to(torch.float32) * 1.0029).to(x.dtype)
            _xf = xf.float()
            xf = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            xf = torch.softmax(xf, dim=-1)
            return F.layer_norm(xf, (xf.shape[-1],), self.ln3_g, self.ln3_b)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        out = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_kernel[(m,)](
            x2, self.rms1_w, self.ln3_g, self.ln3_b, out,
            n, x2.stride(0), out.stride(0),
            RMS_EPS=1e-6, LN_EPS=1e-5,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
