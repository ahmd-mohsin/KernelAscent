import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 694
M, D, DT = 4096, 2048, torch.float16


@triton.jit
def _fused_kernel(
    X, W0, B1, G2, Bt2, B4, Y,
    D_: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D_

    x = tl.load(X + row * D_ + cols, mask=mask, other=0.0)
    x32 = x.to(tl.float32)

    # RMSNorm in fp32, cast to fp16, then fp16 mul by weight
    ms = tl.sum(x32 * x32, axis=0) / D_
    r = 1.0 / tl.sqrt(ms + 1e-6)
    h16 = (x32 * r).to(tl.float16)

    w0 = tl.load(W0 + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    h16 = h16 * w0 + b1  # fp16 arithmetic

    # LayerNorm in fp32
    hf = h16.to(tl.float32)
    hf = tl.where(mask, hf, 0.0)
    mu = tl.sum(hf, axis=0) / D_
    d = tl.where(mask, hf - mu, 0.0)
    var = tl.sum(d * d, axis=0) / D_
    inv = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G2 + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(Bt2 + cols, mask=mask, other=0.0).to(tl.float32)
    y = d * inv * g + b
    y16 = y.to(tl.float16)

    # ReLU + bias in fp16
    zero = tl.zeros_like(y16)
    y16 = tl.maximum(y16, zero)
    b4 = tl.load(B4 + cols, mask=mask, other=0.0)
    y16 = y16 + b4

    tl.store(Y + row * D_ + cols, y16, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.b4 = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = x + self.b1
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = torch.relu(x)
            x = x + self.b4
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        y = torch.empty_like(xc)
        BLOCK = triton.next_power_of_2(d)
        num_warps = 8 if BLOCK >= 2048 else 4
        _fused_kernel[(m,)](
            xc, self.rms0_w, self.b1, self.ln2_g, self.ln2_b, self.b4, y,
            d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
