import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 374
M, D, DT = 2048, 1024, torch.float16


@triton.jit
def _fused_kernel(X, W, G, B, Y,
                  stride_xm, stride_ym,
                  N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x = tl.load(X + row * stride_xm + cols, mask=mask, other=-float('inf')).to(tl.float32)

    # ---- softmax (fp32 accumulate, output cast to fp16) ----
    xmax = tl.max(x, axis=0)
    e = tl.exp(x - xmax)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = (e / s).to(tl.float16)

    # ---- RMSNorm (compute in fp32, cast to fp16, multiply weight in fp16) ----
    xf = sm.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / N
    rrms = 1.0 / tl.sqrt(ms + 1e-6)
    w = tl.load(W + cols, mask=mask, other=0.0)
    h = (xf * rrms).to(tl.float16) * w  # fp16 multiply

    # ---- LayerNorm (fp32 internal, eps=1e-5) ----
    hf = h.to(tl.float32)
    hf = tl.where(mask, hf, 0.0)
    mean = tl.sum(hf, axis=0) / N
    d = tl.where(mask, hf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = (hf - mean) * inv * g + b

    tl.store(Y + row * stride_ym + cols, y.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms1_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)

        orig_shape = x.shape
        x2 = x.contiguous().view(-1, orig_shape[-1])
        m, n = x2.shape
        y = torch.empty_like(x2)
        BLOCK = triton.next_power_of_2(n)
        num_warps = 8 if BLOCK >= 1024 else 4
        _fused_kernel[(m,)](
            x2, self.rms1_w, self.ln2_g, self.ln2_b, y,
            x2.stride(0), y.stride(0),
            N=n, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
