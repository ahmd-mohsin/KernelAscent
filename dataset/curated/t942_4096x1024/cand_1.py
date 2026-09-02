import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 942
M, D, DT = 4096, 1024, torch.float16


@triton.jit
def _fused_rms_relu_ln_gelu(X, W, G, B, Out,
                            D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    # ---- RMSNorm (fp32 math, matching reference) ----
    x = tl.load(X + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * r).to(tl.float16)          # cast to fp16 like .to(x.dtype)

    w = tl.load(W + offs, mask=mask, other=0.0)   # fp16
    y = xn * w                            # fp16 multiply (matches reference)

    # ---- ReLU ----
    y = tl.maximum(y, 0.0)

    # ---- LayerNorm (fp32 accumulation, as PyTorch does for half) ----
    yf = y.to(tl.float32)
    mean = tl.sum(yf, axis=0) / D
    diff = tl.where(mask, yf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    z = diff * rstd * g + b

    # ---- GELU (exact erf, fp32 opmath as PyTorch does for half) ----
    out = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))

    tl.store(Out + row * D + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            x = torch.relu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            return F.gelu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        rows = x2.shape[0]
        out = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8

        _fused_rms_relu_ln_gelu[(rows,)](
            x2, self.rms0_w, self.ln2_g, self.ln2_b, out,
            D=d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return out.view(orig_shape)
