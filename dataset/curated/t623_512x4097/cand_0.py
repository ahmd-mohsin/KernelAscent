import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 623
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_row_kernel(
    X, W1, G, B, W4, Y,
    D_: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    # ---- load + scalar mul (fp32 opmath, round to fp16 like PyTorch) ----
    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * 1.3301).to(tl.float16).to(tl.float32)

    # ---- RMSNorm 1 ----
    ms = tl.sum(x * x, axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)
    t = (x * r).to(tl.float16).to(tl.float32)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (t * w1).to(tl.float16).to(tl.float32)

    # ---- LayerNorm (fp32 accumulation, fp16 output) ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D_
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D_
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    x = ((x - mean) * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---- Softmax (fp32 internal, fp16 output) ----
    xm = tl.where(mask, x, float('-inf'))
    mx = tl.max(xm, axis=0)
    e = tl.exp(xm - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16).to(tl.float32)

    # ---- RMSNorm 2 ----
    ms2 = tl.sum(x * x, axis=0) / D_
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    t2 = (x * r2).to(tl.float16).to(tl.float32)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (t2 * w4).to(tl.float16)

    tl.store(Y + row * D_ + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x * 1.3301
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = torch.softmax(x, dim=-1)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_row_kernel[(m,)](
            x2, self.rms1_w, self.ln2_g, self.ln2_b, self.rms4_w, y,
            D_=d, BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view(orig_shape)
