import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 562
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _fused_kernel(
    X, W0, B1, G2, B2, W3, OUT,
    n_cols,
    stride_x, stride_out,
    eps_rms, eps_ln,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 0 (fp32 math, cast to fp16, fp16 multiply by weight) ----
    ms = tl.sum(x * x, axis=0) / n_cols
    r0 = 1.0 / tl.sqrt(ms + eps_rms)
    y16 = (x * r0).to(tl.float16)

    w0 = tl.load(W0 + cols, mask=mask, other=0.0)  # fp16
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)  # fp16
    y16 = y16 * w0        # fp16 multiply (matches PyTorch half elementwise)
    y16 = y16 + b1        # fp16 add

    # ---- LayerNorm (fp32 accumulation, matches PyTorch half layer_norm) ----
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    mean = tl.sum(yf, axis=0) / n_cols
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    inv = 1.0 / tl.sqrt(var + eps_ln)

    g2 = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    z = (yf - mean) * inv * g2 + b2
    z16 = z.to(tl.float16)

    # ---- RMSNorm 3 ----
    zf = z16.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    ms2 = tl.sum(zf * zf, axis=0) / n_cols
    r3 = 1.0 / tl.sqrt(ms2 + eps_rms)
    u16 = (zf * r3).to(tl.float16)

    w3 = tl.load(W3 + cols, mask=mask, other=0.0)  # fp16
    u16 = u16 * w3  # fp16 multiply

    # ---- GELU (exact erf, computed in fp32 like PyTorch opmath) ----
    uf = u16.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865475
    gout = uf * 0.5 * (1.0 + tl.math.erf(uf * INV_SQRT2))
    out16 = gout.to(tl.float16)

    tl.store(OUT + row * stride_out + cols, out16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda or x.dtype != torch.float16:
            return self._ref_forward(x)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_kernel[(n_rows,)](
            x2d, self.rms0_w, self.b1, self.ln2_g, self.ln2_b, self.rms3_w, out,
            n_cols,
            x2d.stride(0), out.stride(0),
            1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)

    def _ref_forward(self, x):
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = x + self.b1
        x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        x = F.gelu(x)
        return x
