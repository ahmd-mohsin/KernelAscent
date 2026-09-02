import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 901
M, D, DT = 4096, 2049, torch.float16


@triton.jit
def _fused_norm_chain(
    X, Y,
    G0, B0, W1, G2, B2, G3, B3, G4, B4,
    N, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    Nf = N.to(tl.float32)

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- LayerNorm 0 ----
    mean = tl.sum(x, 0) / Nf
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, 0) / Nf
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G0 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B0 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xm * rstd * g + b).to(tl.float16).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    # ---- RMSNorm 1 (mul by weight in fp16, matching reference) ----
    ms = tl.sum(x * x, 0) / Nf
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xh = (x * r).to(tl.float16)
    w = tl.load(W1 + offs, mask=mask, other=0.0)
    x = (xh * w).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    # ---- LayerNorm 2 ----
    mean = tl.sum(x, 0) / Nf
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, 0) / Nf
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xm * rstd * g + b).to(tl.float16).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    # ---- LayerNorm 3 ----
    mean = tl.sum(x, 0) / Nf
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, 0) / Nf
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G3 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B3 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xm * rstd * g + b).to(tl.float16).to(tl.float32)
    x = tl.where(mask, x, 0.0)

    # ---- LayerNorm 4 ----
    mean = tl.sum(x, 0) / Nf
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, 0) / Nf
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G4 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xm * rstd * g + b).to(tl.float16)

    tl.store(Y + row * stride_y + offs, y, mask=mask)


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
        if not x.is_cuda or x.dtype != torch.float16:
            # fallback (reference path)
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
        num_warps = 8 if BLOCK <= 4096 else 16

        _fused_norm_chain[(rows,)](
            x2, y,
            self.ln0_g, self.ln0_b, self.rms1_w,
            self.ln2_g, self.ln2_b,
            self.ln3_g, self.ln3_b,
            self.ln4_g, self.ln4_b,
            N, x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)


def get_inputs():
    g = torch.Generator().manual_seed(SEED + 12345)
    return [torch.randn(M, D, generator=g).to(DT)]
