import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 505
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, OUT, W0, W2, G3, B3,
    stride_xm, stride_om,
    N, RMS_EPS, LN_EPS,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # --- RMSNorm 0 ---
    ms = tl.sum(xf * xf, axis=0) / N
    rr = 1.0 / tl.sqrt(ms + RMS_EPS)
    t = (xf * rr).to(tl.bfloat16)                      # round to bf16
    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    t = (t.to(tl.float32) * w0.to(tl.float32)).to(tl.bfloat16)

    # --- ReLU ---
    t = tl.maximum(t, 0.0)

    # --- RMSNorm 2 ---
    tf = t.to(tl.float32)
    ms2 = tl.sum(tf * tf, axis=0) / N
    rr2 = 1.0 / tl.sqrt(ms2 + RMS_EPS)
    t = (tf * rr2).to(tl.bfloat16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    t = (t.to(tl.float32) * w2.to(tl.float32)).to(tl.bfloat16)

    # --- LayerNorm ---
    tf = t.to(tl.float32)
    mean = tl.sum(tl.where(mask, tf, 0.0), axis=0) / N
    diff = tl.where(mask, tf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + LN_EPS)
    g = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    y = (diff * rstd * g + b).to(tl.bfloat16)

    tl.store(OUT + row * stride_om + cols, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.relu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)

        orig_shape = x.shape
        N = orig_shape[-1]
        x2d = x.contiguous().view(-1, N)
        Mrows = x2d.shape[0]
        out = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(Mrows,)](
            x2d, out,
            self.rms0_w, self.rms2_w, self.ln3_g, self.ln3_b,
            x2d.stride(0), out.stride(0),
            N, 1e-6, 1e-5,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(orig_shape)
