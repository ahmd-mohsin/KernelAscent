import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 626
M, D, DT = 2048, 513, torch.float16


@triton.jit
def _fused_row_kernel(
    X, W, G, B, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- scale: x = x * 1.0029 (fp16 tensor * python scalar -> opmath fp32, round fp16)
    x16 = tl.load(X + base + offs, mask=mask, other=0.0)
    xf = x16.to(tl.float32) * 1.0029
    xh = xf.to(tl.float16)

    # ---- RMSNorm in fp32, cast to fp16, then * weight
    xf2 = xh.to(tl.float32)
    sq = tl.where(mask, xf2 * xf2, 0.0)
    ms = tl.sum(sq, axis=0) / D
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    r16 = (xf2 * inv).to(tl.float16)

    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    t16 = (r16.to(tl.float32) * w).to(tl.float16)

    # ---- softmax (fp32 accumulation, fp16 output)
    tf = t16.to(tl.float32)
    tf_masked = tl.where(mask, tf, float('-inf'))
    mx = tl.max(tf_masked, axis=0)
    e = tl.exp(tf_masked - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    p16 = (e / s).to(tl.float16)

    # ---- LayerNorm (fp32 stats, eps=1e-5, fp16 output)
    pf = p16.to(tl.float32)
    pm = tl.where(mask, pf, 0.0)
    mean = tl.sum(pm, axis=0) / D
    d = tl.where(mask, pf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((pf - mean) * rstd * g + b).to(tl.float16)
    tl.store(Y + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = x * 1.0029
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = torch.softmax(x, dim=-1)
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return x

        orig_shape = x.shape
        Dn = orig_shape[-1]
        xc = x.contiguous().view(-1, Dn)
        Mn = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(Dn)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_row_kernel[(Mn,)](
            xc, self.rms1_w, self.ln3_g, self.ln3_b, y,
            D=Dn, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
