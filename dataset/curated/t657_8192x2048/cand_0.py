import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 657
M, D, DT = 8192, 2048, torch.bfloat16


@triton.jit
def _fused_rms2_gelu2_kernel(
    X, W0, W1, Y,
    n_rows,
    eps,
    D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    ptr = row * D + offs

    # ---- RMSNorm 0 ----
    xf = tl.load(X + ptr).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / D + eps)
    xb = (xf * r).to(tl.bfloat16)                       # round like .to(x.dtype)
    w0 = tl.load(W0 + offs).to(tl.float32)
    xb = (xb.to(tl.float32) * w0).to(tl.bfloat16)       # bf16 multiply semantics

    # ---- RMSNorm 1 ----
    xf = xb.to(tl.float32)
    r = tl.math.rsqrt(tl.sum(xf * xf, axis=0) / D + eps)
    xb = (xf * r).to(tl.bfloat16)
    w1 = tl.load(W1 + offs).to(tl.float32)
    xb = (xb.to(tl.float32) * w1).to(tl.bfloat16)

    # ---- GELU (exact, erf) x2 with bf16 rounding between ----
    INV_SQRT2: tl.constexpr = 0.7071067811865476
    xf = xb.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    xb = g.to(tl.bfloat16)

    xf = xb.to(tl.float32)
    g = 0.5 * xf * (1.0 + tl.math.erf(xf * INV_SQRT2))
    tl.store(Y + ptr, g.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            return F.gelu(F.gelu(x))

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        y = torch.empty_like(x2)

        _fused_rms2_gelu2_kernel[(n_rows,)](
            x2, self.rms0_w, self.rms1_w, y,
            n_rows, 1e-6,
            D=d,
            num_warps=8,
        )
        return y.view(orig_shape)
