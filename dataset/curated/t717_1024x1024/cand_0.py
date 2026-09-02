import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 717
M, D, DT = 1024, 1024, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, Y,
    LN0G, LN0B, LN1G, LN1B, RMS2W, LN3G, LN3B, RMS4W,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    g = tl.load(LN0G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN0B + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    y = xc * tl.rsqrt(var + 1e-5) * g + b
    x = y.to(tl.float16).to(tl.float32)  # round to fp16 like reference

    # ---- LayerNorm 1 ----
    g = tl.load(LN1G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN1B + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    y = xc * tl.rsqrt(var + 1e-5) * g + b
    x = y.to(tl.float16).to(tl.float32)

    # ---- RMSNorm 2 ----
    ms = tl.sum(x * x, axis=0) / D
    xh = (x * tl.rsqrt(ms + 1e-6)).to(tl.float16)
    w = tl.load(RMS2W + offs, mask=mask, other=0.0)  # fp16
    x = (xh * w).to(tl.float32)  # fp16 multiply as in reference

    # ---- LayerNorm 3 ----
    g = tl.load(LN3G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LN3B + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    y = xc * tl.rsqrt(var + 1e-5) * g + b
    x = y.to(tl.float16).to(tl.float32)

    # ---- RMSNorm 4 ----
    ms = tl.sum(x * x, axis=0) / D
    xh = (x * tl.rsqrt(ms + 1e-6)).to(tl.float16)
    w = tl.load(RMS4W + offs, mask=mask, other=0.0)  # fp16
    out = xh * w  # fp16 multiply

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference implementation
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2d = x.contiguous().view(-1, d)
        rows = x2d.shape[0]
        y = torch.empty_like(x2d)
        BLOCK = triton.next_power_of_2(d)
        _fused_norm_kernel[(rows,)](
            x2d, y,
            self.ln0_g, self.ln0_b,
            self.ln1_g, self.ln1_b,
            self.rms2_w,
            self.ln3_g, self.ln3_b,
            self.rms4_w,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
