import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 695
M, D, DT = 2048, 1024, torch.bfloat16


@triton.jit
def _fused_softmax_rms_kernel(
    X, W2, W4, Y,
    D: tl.constexpr,
    S1: tl.constexpr,   # 1.4612
    S2: tl.constexpr,   # 1.1354
    EPS: tl.constexpr,  # 1e-6
):
    row = tl.program_id(0)
    offs = tl.arange(0, D)
    base = row * D

    # ---- scale (emulate bf16 rounding of x * 1.4612) ----
    x = tl.load(X + base + offs).to(tl.float32)
    x = (x * S1).to(tl.bfloat16).to(tl.float32)

    # ---- softmax (float accumulation, bf16 output like torch) ----
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    p = (e / s).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm #1 (fp32 compute, bf16 round, bf16*bf16 -> bf16) ----
    ms1 = tl.sum(p * p, axis=0) / D
    r1 = tl.math.rsqrt(ms1 + EPS)
    y = (p * r1).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + offs).to(tl.float32)
    y = (y * w2).to(tl.bfloat16).to(tl.float32)

    # ---- scale (bf16 round) ----
    y = (y * S2).to(tl.bfloat16).to(tl.float32)

    # ---- RMSNorm #2 ----
    ms2 = tl.sum(y * y, axis=0) / D
    r2 = tl.math.rsqrt(ms2 + EPS)
    z = (y * r2).to(tl.bfloat16).to(tl.float32)
    w4 = tl.load(W4 + offs).to(tl.float32)
    z = (z * w4).to(tl.bfloat16)

    tl.store(Y + base + offs, z)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms2_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(1024, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # reference path for CPU
            x = x * 1.4612
            x = torch.softmax(x, dim=-1)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = x * 1.1354
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms4_w
            return x

        orig_shape = x.shape
        d = orig_shape[-1]
        x2 = x.contiguous().view(-1, d)
        n_rows = x2.shape[0]
        out = torch.empty_like(x2)

        _fused_softmax_rms_kernel[(n_rows,)](
            x2, self.rms2_w, self.rms4_w, out,
            D=d,
            S1=1.4612,
            S2=1.1354,
            EPS=1e-6,
            num_warps=8,
        )
        return out.view(orig_shape)
