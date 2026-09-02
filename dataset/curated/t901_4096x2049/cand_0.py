import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 901
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _fused_norms_kernel(
    X, Y,
    G0, B0, W1, G2, B2, G3, B3, G4, B4,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    Nf = N.to(tl.float32)

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)

    # ---------- LayerNorm 0 (fp32 compute, cast result to fp16) ----------
    g = tl.load(G0 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B0 + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / Nf
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / Nf
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    x = (xc * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---------- RMSNorm (fp32 norm, cast to fp16, multiply weight in fp16) ----------
    ms = tl.sum(x * x, axis=0) / Nf
    rr = 1.0 / tl.sqrt(ms + 1e-6)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)  # fp16
    x16 = (x * rr).to(tl.float16) * w1              # fp16 multiply, matches reference
    x = x16.to(tl.float32)

    # ---------- LayerNorm 2 ----------
    g = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / Nf
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / Nf
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    x = (xc * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---------- LayerNorm 3 ----------
    g = tl.load(G3 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / Nf
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / Nf
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    x = (xc * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---------- LayerNorm 4 ----------
    g = tl.load(G4 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4 + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / Nf
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / Nf
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    out = (xc * rstd * g + b).to(tl.float16)

    tl.store(Y + row * stride_y + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln0_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln0_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2049, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = F.layer_norm(x, (x.shape[-1],), self.ln0_g, self.ln0_b)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x

        orig_shape = x.shape
        N = orig_shape[-1]
        x2 = x.contiguous().view(-1, N)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(N)
        _fused_norms_kernel[(rows,)](
            x2, y,
            self.ln0_g, self.ln0_b, self.rms1_w,
            self.ln2_g, self.ln2_b,
            self.ln3_g, self.ln3_b,
            self.ln4_g, self.ln4_b,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
