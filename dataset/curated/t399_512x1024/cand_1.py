import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 399
M, D, DT = 512, 1024, torch.float16


@triton.jit
def _fused_bias_ln_rms(Y, B1, G, Bt, Wr, Out, N, stride, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    y = tl.load(Y + row * stride + cols, mask=mask, other=0.0)
    b1 = tl.load(B1 + cols, mask=mask, other=0.0)
    # bias add in fp16 (matches reference x + b1)
    x = (y + b1).to(tl.float16)

    # LayerNorm in fp32 (matches PyTorch internal upcast)
    xf = x.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / N
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    bt = tl.load(Bt + cols, mask=mask, other=0.0).to(tl.float32)
    ln = ((xf - mean) * inv * g + bt).to(tl.float16)

    # RMSNorm: fp32 compute on fp16 values, cast to fp16, multiply by w in fp16
    lf = ln.to(tl.float32)
    ms = tl.sum(tl.where(mask, lf * lf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + 1e-6)
    rn = (lf * r).to(tl.float16)

    w = tl.load(Wr + cols, mask=mask, other=0.0)
    out = rn * w
    tl.store(Out + row * stride + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        y = x @ self.W0
        y = y.contiguous()
        Mrows, N = y.shape
        out = torch.empty_like(y)
        _fused_bias_ln_rms[(Mrows,)](
            y, self.b1, self.ln2_g, self.ln2_b, self.rms3_w, out,
            N, y.stride(0), BLOCK=512, num_warps=4,
        )
        return out
