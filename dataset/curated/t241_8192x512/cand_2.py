import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 241
M, D, DT = 8192, 512, torch.float16


@triton.jit
def _fused_rms_softmax_ln_kernel(
    X, W, G, B, OUT,
    stride_x, stride_o,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x16 = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    x32 = x16.to(tl.float32)

    # ---- RMSNorm (fp32 math, cast to fp16, multiply by fp16 weight) ----
    ms = tl.sum(x32 * x32, axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)
    y16 = (x32 * r).to(tl.float16)
    w16 = tl.load(W + cols, mask=mask, other=0.0)
    y16 = y16 * w16  # fp16 multiply, matches reference

    # ---- Softmax (fp32 accumulation, fp16 output) ----
    y32 = y16.to(tl.float32)
    y32m = tl.where(mask, y32, float("-inf"))
    mx = tl.max(y32m, axis=0)
    e = tl.exp(y32m - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p16 = (e / s).to(tl.float16)

    # ---- LayerNorm (fp32 internal math, fp16 output) ----
    p32 = p16.to(tl.float32)
    mean = tl.sum(tl.where(mask, p32, 0.0), axis=0) / D_
    diff = tl.where(mask, p32 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D_
    inv = 1.0 / tl.sqrt(var + 1e-5)

    g32 = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b32 = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = (diff * inv * g32 + b32).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.softmax(x, dim=-1)
            return F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_rms_softmax_ln_kernel[(m,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, out,
            x2.stride(0), out.stride(0),
            D_=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
