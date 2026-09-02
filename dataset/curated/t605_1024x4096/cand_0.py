import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 605
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _softmax_rmsnorm_kernel(
    X, W, Y,
    stride_xm, stride_ym,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * stride_xm + offs, mask=mask, other=float('-inf')).to(tl.float32)

    # softmax (fp32 accumulate, matching torch.softmax on fp16)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = e / s

    # cast to fp16 (as torch.softmax returns fp16), then back to fp32 for RMS
    p16 = p.to(tl.float16)
    pf = p16.to(tl.float32)

    ms = tl.sum(tl.where(mask, pf * pf, 0.0), axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)

    w = tl.load(W + offs, mask=mask, other=0.0)
    y = (pf * r).to(tl.float16) * w

    tl.store(Y + row * stride_ym + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 4096, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        Mrows, Dcols = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(Dcols)
        _softmax_rmsnorm_kernel[(Mrows,)](
            x, self.rms2_w, y,
            x.stride(0), y.stride(0),
            Dcols, BLOCK=BLOCK,
            num_warps=8,
        )
        return y
