import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 602
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(X, W, B, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_
    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0).to(tl.float32)

    # RMS norm (fp32)
    ms = tl.sum(x * x, axis=0) / D_
    rinv = 1.0 / tl.sqrt(ms + 1e-6)

    # (xf * rsqrt).to(bf16)
    t = (x * rinv).to(tl.bfloat16).to(tl.float32)

    # * rms0_w (bf16 op, fp32 opmath, round to bf16)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    t = (t * w).to(tl.bfloat16).to(tl.float32)

    # + b1
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    t = (t + b).to(tl.bfloat16).to(tl.float32)

    # gelu (erf-based, fp32 opmath, round to bf16)
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))
    t = g.to(tl.bfloat16).to(tl.float32)

    # * 1.3699 then * 1.2531, rounding to bf16 each time
    t = (t * 1.3699).to(tl.bfloat16).to(tl.float32)
    t = (t * 1.2531).to(tl.bfloat16)

    tl.store(Y + row * D_ + offs, t, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            y = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            y = y + self.b1
            y = F.gelu(y)
            y = y * 1.3699
            y = y * 1.2531
            return y

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.rms0_w, self.b1, y,
            D_=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
