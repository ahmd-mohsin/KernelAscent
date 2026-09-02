import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 641
M, D, DT = 4096, 512, torch.bfloat16


@triton.jit
def _fused_kernel(X, W, Y, D_: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm (mean over full D)
    ms = tl.sum(x * x, axis=0) / D_
    inv = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * inv).to(tl.bfloat16)

    w = tl.load(W + offs, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(tl.bfloat16)

    # GELU (exact, erf), computed in fp32, cast back to bf16 each time
    yf = y.to(tl.float32)
    yf = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    y = yf.to(tl.bfloat16)

    yf = y.to(tl.float32)
    yf = yf * 0.5 * (1.0 + tl.math.erf(yf * 0.7071067811865476))
    y = yf.to(tl.bfloat16)

    # Softmax in fp32
    yf = y.to(tl.float32)
    yf = tl.where(mask, yf, float('-inf'))
    m = tl.max(yf, axis=0)
    e = tl.exp(yf - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    out = (e / s).to(tl.bfloat16)

    tl.store(Y + row * D_ + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        rows = x.numel() // x.shape[-1]
        d = x.shape[-1]
        y = torch.empty_like(x)
        _fused_kernel[(rows,)](
            x, self.rms0_w, y,
            D_=d, BLOCK=triton.next_power_of_2(d),
            num_warps=4,
        )
        return y
