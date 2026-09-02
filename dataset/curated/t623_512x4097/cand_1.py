import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 623
M, D, DT = 512, 4097, torch.float16


@triton.jit
def _fused_kernel(
    X, W1, G, B, W4, Y,
    D_, stride_x, stride_y,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_
    d_f = D_.to(tl.float32)

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # scale (opmath fp32 -> fp16)
    x = (x * 1.3301).to(tl.float16)

    # RMSNorm 1
    xf = x.to(tl.float32)
    ms = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / d_f
    r = 1.0 / tl.sqrt(ms + 1e-6)
    x16 = (xf * r).to(tl.float16)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x16.to(tl.float32) * w1).to(tl.float16)

    # LayerNorm
    xf = x.to(tl.float32)
    mean = tl.sum(tl.where(mask, xf, 0.0), axis=0) / d_f
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / d_f
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    x = ((xf - mean) * inv * g + b).to(tl.float16)

    # Softmax (fp32 accumulate)
    xf = tl.where(mask, x.to(tl.float32), float('-inf'))
    mx = tl.max(xf, axis=0)
    e = tl.exp(xf - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    x = (e / s).to(tl.float16)

    # RMSNorm 2
    xf = x.to(tl.float32)
    ms2 = tl.sum(tl.where(mask, xf * xf, 0.0), axis=0) / d_f
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    x16 = (xf * r2).to(tl.float16)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (x16.to(tl.float32) * w4).to(tl.float16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4097, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(d)
        _fused_kernel[(m,)](
            x2, self.rms1_w, self.ln2_g, self.ln2_b, self.rms4_w, y,
            d, x2.stride(0), y.stride(0),
            BLOCK=BLOCK,
            num_warps=16,
        )
        return y.view(orig_shape)
