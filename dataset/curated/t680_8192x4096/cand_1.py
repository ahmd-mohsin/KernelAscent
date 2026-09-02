import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 680
M, D, DT = 8192, 4096, torch.float16


@triton.jit
def _softmax_rms_rms_kernel(
    X, W1, W3, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- softmax (fp32 accumulation, matching torch's half softmax) ----
    x = tl.load(X + row * D + offs, mask=mask, other=float('-inf')).to(tl.float32)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(tl.where(mask, e, 0.0), axis=0)
    p16 = (e / s).to(tl.float16)  # round to fp16 like reference output of softmax

    # ---- RMSNorm 1 ----
    pf = p16.to(tl.float32)
    ms1 = tl.sum(tl.where(mask, pf * pf, 0.0), axis=0) / D
    r1 = tl.rsqrt(ms1 + 1e-6)
    h16 = (pf * r1).to(tl.float16)                     # .to(x.dtype)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    h16 = (h16.to(tl.float32) * w1.to(tl.float32)).to(tl.float16)  # * rms1_w (fp16 mul, fp32 opmath)

    # ---- scalar scale ----
    h16 = (h16.to(tl.float32) * 1.4345).to(tl.float16)

    # ---- RMSNorm 2 ----
    hf = h16.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, hf * hf, 0.0), axis=0) / D
    r2 = tl.rsqrt(ms2 + 1e-6)
    g16 = (hf * r2).to(tl.float16)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    out = (g16.to(tl.float32) * w3.to(tl.float32)).to(tl.float16)

    tl.store(Y + row * D + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            xx = torch.softmax(x, dim=-1)
            _xf = xx.float()
            xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms1_w
            xx = xx * 1.4345
            _xf = xx.float()
            xx = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(xx.dtype) * self.rms3_w
            return xx

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        rows = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _softmax_rms_rms_kernel[(rows,)](
            xc, self.rms1_w, self.rms3_w, y,
            D=d, BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return y.view(orig_shape)
