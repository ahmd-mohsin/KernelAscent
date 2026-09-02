import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 770
M, D, DT = 4096, 1024, torch.bfloat16


@triton.jit
def _fused_kernel(
    X, B0, W1, G2, B2, Y,
    D_: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_

    x = tl.load(X + row * D_ + offs, mask=mask, other=0.0)
    b0 = tl.load(B0 + offs, mask=mask, other=0.0)

    # x = x + b0 (bf16 add)
    x = (x + b0).to(tl.bfloat16)

    # RMSNorm in fp32, cast to bf16, multiply by w in bf16
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D_
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0)
    xr = (xf * rrms).to(tl.bfloat16)
    x2 = (xr * w1).to(tl.bfloat16)

    # LayerNorm: compute in fp32 (as PyTorch does for bf16 input)
    t = x2.to(tl.float32)
    mean = tl.sum(tl.where(mask, t, 0.0), axis=0) / D_
    d = tl.where(mask, t - mean, 0.0)
    var = tl.sum(d * d, axis=0) / D_
    invstd = 1.0 / tl.sqrt(var + 1e-5)

    g2 = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b2 = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (t - mean) * invstd * g2 + b2

    tl.store(Y + row * D_ + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.b0 = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = x + self.b0
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        y = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            xc, self.b0, self.rms1_w, self.ln2_g, self.ln2_b, y,
            D_=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
