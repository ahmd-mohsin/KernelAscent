import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 505
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_rms_relu_rms_ln_kernel(
    X, W0, W2, G, B, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- RMSNorm 0 ----
    x = tl.load(X + base + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    ms = tl.sum(xf * xf, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xb = (xf * r).to(tl.bfloat16)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0)
    xb = xb * w0

    # ---- ReLU (exact on bf16: no rounding introduced) ----
    xb = tl.maximum(xb.to(tl.float32), 0.0).to(tl.bfloat16)

    # ---- RMSNorm 2 ----
    xf = xb.to(tl.float32)
    ms2 = tl.sum(xf * xf, axis=0) / D
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    xb = (xf * r2).to(tl.bfloat16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    xb = xb * w2

    # ---- LayerNorm (fp32 internal math, as PyTorch does for bf16) ----
    xf = xb.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / D
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = diff * inv * g + b
    tl.store(Y + base + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            return self._forward_ref(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        xc = x.contiguous().view(-1, d)
        m = xc.shape[0]
        y = torch.empty_like(xc)

        BLOCK = triton.next_power_of_2(d)
        _fused_rms_relu_rms_ln_kernel[(m,)](
            xc, self.rms0_w, self.rms2_w, self.ln3_g, self.ln3_b, y,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)

    def _forward_ref(self, x):
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = torch.relu(x)
        _xf = x.float()
        x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
        return x
