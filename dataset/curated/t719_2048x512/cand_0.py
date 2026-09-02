import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 719
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _fused_row_kernel(
    X_ptr, G_ptr, B_ptr, W_ptr, Y_ptr,
    D: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D

    x = tl.load(X_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- GELU (exact, erf-based; opmath fp32, round to bf16) ----
    x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- LayerNorm (fp32 internal, round to bf16) ----
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / D
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / D
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x = xc * rstd * g + b
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- Softmax (fp32 internal, round to bf16) ----
    mval = tl.max(tl.where(mask, x, float('-inf')), axis=0)
    e = tl.where(mask, tl.exp(x - mval), 0.0)
    s = tl.sum(e, axis=0)
    x = e / s
    x = x.to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm part: (xf * rsqrt(mean(xf^2)+1e-6)).to(bf16) ----
    ms = tl.sum(tl.where(mask, x * x, 0.0), axis=0) / D
    r = 1.0 / tl.sqrt(ms + 1e-6)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)

    # ---- * rms3_w (bf16 elementwise mul, opmath fp32) then ReLU ----
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xn * w).to(tl.bfloat16).to(tl.float32)
    y = tl.maximum(y, 0.0)

    tl.store(Y_ptr + row * D + offs, y.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.ln1_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback (reference path)
            x = F.gelu(x)
            x = F.layer_norm(x, (x.shape[-1],), self.ln1_g, self.ln1_b)
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
            return torch.relu(x)

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        m = x2.shape[0]
        y = torch.empty_like(x2)

        BLOCK = triton.next_power_of_2(d)
        num_warps = 4 if BLOCK <= 1024 else 8
        _fused_row_kernel[(m,)](
            x2, self.ln1_g, self.ln1_b, self.rms3_w, y,
            D=d, BLOCK=BLOCK, num_warps=num_warps,
        )
        return y.view(orig_shape)
