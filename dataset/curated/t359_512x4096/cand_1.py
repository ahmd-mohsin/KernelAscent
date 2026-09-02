import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 359
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, OUT,
    G0, B0, W2, G3, B3,
    N,
    stride_x, stride_o,
    LN_EPS: tl.constexpr,
    RMS_EPS: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 (fp32 math, bf16 output) ----
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    inv = 1.0 / tl.sqrt(var + LN_EPS)
    g0 = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b0 = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (xc * inv * g0 + b0).to(tl.bfloat16)

    # ---- scale by S1 (opmath fp32, bf16 output) ----
    y = (y.to(tl.float32) * S1).to(tl.bfloat16)

    # ---- RMSNorm (fp32 math, bf16 output, then * w2 in fp32 -> bf16) ----
    xf = y.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + RMS_EPS)
    z = (xf * r).to(tl.bfloat16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    z = (z.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # ---- LayerNorm 3 (fp32 math, bf16 output) ----
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    mean3 = tl.sum(zf, axis=0) / N
    zc = tl.where(mask, zf - mean3, 0.0)
    var3 = tl.sum(zc * zc, axis=0) / N
    inv3 = 1.0 / tl.sqrt(var3 + LN_EPS)
    g3 = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b3 = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    o = (zc * inv3 * g3 + b3).to(tl.bfloat16)

    # ---- scale by S2 ----
    o = (o.to(tl.float32) * S2).to(tl.bfloat16)

    tl.store(OUT + row * stride_o + cols, o, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            y = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            y = y * 1.0166
            _xf = y.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(y.dtype) * self.rms2_w
            y = F.layer_norm(y, (y.shape[-1],), self.ln3_g, self.ln3_b)
            return y * 1.0289

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        rows = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_norm_kernel[(rows,)](
            x2d, out,
            self.ln0_g, self.ln0_b, self.rms2_w, self.ln3_g, self.ln3_b,
            N,
            x2d.stride(0), out.stride(0),
            LN_EPS=1e-5,
            RMS_EPS=1e-6,
            S1=1.0166,
            S2=1.0289,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
