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
    X, OUT, W0, B1, G2, B2, W3,
    n_cols, stride_x, stride_o,
    eps_rms, eps_ln,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # ---- RMSNorm 0 (fp32 compute, cast to fp16, fp16 multiply by weight) ----
    ms0 = tl.sum(xf * xf, axis=0) / n_cols
    r0 = 1.0 / tl.sqrt(ms0 + eps_rms)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0)
    h = (xf * r0).to(tl.float16) * w0  # fp16 arithmetic

    # ---- bias add (fp16) ----
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)
    h = h + b1

    # ---- LayerNorm (fp32 compute) ----
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, 0.0)
    mean = tl.sum(hf, axis=0) / n_cols
    diff = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps_ln)
    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd * g2 + b2).to(tl.float16)

    # ---- RMSNorm 3 ----
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    ms3 = tl.sum(yf * yf, axis=0) / n_cols
    r3 = 1.0 / tl.sqrt(ms3 + eps_rms)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    z = (yf * r3).to(tl.float16) * w3  # fp16 arithmetic

    # ---- GELU (exact, fp32 compute like PyTorch opmath) ----
    zf = z.to(tl.float32)
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    out = 0.5 * zf * (1.0 + tl.math.erf(zf * INV_SQRT2))
    out = out.to(tl.float16)

    tl.store(OUT + row * stride_o + offs, out, mask=mask)


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
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x + self.b1
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return F.gelu(x)

        orig_shape = x.shape
        n_cols = orig_shape[-1]
        x2d = x.contiguous().view(-1, n_cols)
        n_rows = x2d.shape[0]
        out = torch.empty_like(x2d)

        BLOCK = triton.next_power_of_2(n_cols)
        num_warps = 8 if BLOCK >= 1024 else 4

        _fused_kernel[(n_rows,)](
            x2d, out,
            self.rms0_w, self.b1, self.ln2_g, self.ln2_b, self.rms3_w,
            n_cols, x2d.stride(0), out.stride(0),
            1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
