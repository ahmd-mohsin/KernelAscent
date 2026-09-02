import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 505
M, D, DT = 1024, 2048, torch.bfloat16


@triton.jit
def _fused_norm_kernel(
    X, W0, W2, G, B, Y,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D
    base = row * D

    # ---- RMSNorm 0 (compute in fp32, round to bf16, mul by weight in bf16) ----
    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)
    r0 = tl.math.rsqrt(tl.sum(x * x, axis=0) / D + 1e-6)
    xb = (x * r0).to(tl.bfloat16)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0)
    xb = (xb * w0).to(tl.bfloat16)

    # ---- ReLU (exact) ----
    xb = tl.where(xb > 0, xb, 0.0).to(tl.bfloat16)

    # ---- RMSNorm 2 ----
    xf = xb.to(tl.float32)
    r2 = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / D + 1e-6)
    xb = (xf * r2).to(tl.bfloat16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0)
    xb = (xb * w2).to(tl.bfloat16)

    # ---- LayerNorm (fp32 accumulation, like PyTorch CUDA kernel) ----
    xf = xb.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    mean = tl.sum(xf, axis=0) / D
    diff = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = ((xf - mean) * inv * g + b).to(tl.bfloat16)

    tl.store(Y + base + offs, y, mask=mask)


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
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.relu(x)
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        _fused_norm_kernel[(rows,)](
            x2, self.rms0_w, self.rms2_w, self.ln3_g, self.ln3_b, y,
            D=d, BLOCK=BLOCK,
            num_warps=8,
        )
        return y.view(orig_shape)
