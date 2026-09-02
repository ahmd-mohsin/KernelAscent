import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 662
M, D, DT = 512, 513, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, W0, B1, W2, Y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0)
    xf = x.to(tl.float32)

    # RMSNorm 0 (mean of squares in fp32)
    ms0 = tl.sum(xf * xf, axis=0) / D_
    inv0 = 1.0 / tl.sqrt(ms0 + 1e-6)
    h = (xf * inv0).to(tl.bfloat16)

    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    h = h * w0 + b1  # bf16 arithmetic (matches reference)

    # RMSNorm 2
    hf = h.to(tl.float32)
    hf_masked = tl.where(mask, hf, 0.0)
    ms1 = tl.sum(hf_masked * hf_masked, axis=0) / D_
    inv1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    g = (hf * inv1).to(tl.bfloat16)

    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    g = g * w2  # bf16 multiply (matches reference)

    # exact GELU (erf), computed in fp32 like PyTorch's opmath
    gf = g.to(tl.float32)
    out = 0.5 * gf * (1.0 + tl.math.erf(gf * 0.7071067811865476))

    tl.store(Y + row * D_ + cols, out.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(513, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x + self.b1
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            return F.gelu(x)

        x = x.contiguous()
        rows = x.numel() // x.shape[-1]
        d = x.shape[-1]
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(rows,)](
            x, self.rms0_w, self.b1, self.rms2_w, y,
            D_=d, BLOCK=BLOCK,
            num_warps=8 if BLOCK >= 1024 else 4,
        )
        return y
