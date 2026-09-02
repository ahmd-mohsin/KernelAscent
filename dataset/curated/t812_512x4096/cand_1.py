import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 812
M, D, DT = 512, 4096, torch.bfloat16


@triton.jit
def _triple_rmsnorm_kernel(
    X, W0, W2, W3, Y,
    D_size,
    scale,
    eps,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < D_size
    base = row * D_size

    x = tl.load(X + base + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- stage 0: rmsnorm + w0 ----
    ms = tl.sum(x * x, axis=0) / D_size
    r = tl.math.rsqrt(ms + eps)
    xn = (x * r).to(tl.bfloat16).to(tl.float32)
    w0 = tl.load(W0 + offs, mask=mask, other=0.0).to(tl.float32)
    x1 = (xn * w0).to(tl.bfloat16).to(tl.float32)

    # ---- scale ----
    x1 = (x1 * scale).to(tl.bfloat16).to(tl.float32)

    # ---- stage 2: rmsnorm + w2 ----
    ms = tl.sum(x1 * x1, axis=0) / D_size
    r = tl.math.rsqrt(ms + eps)
    xn = (x1 * r).to(tl.bfloat16).to(tl.float32)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    x2 = (xn * w2).to(tl.bfloat16).to(tl.float32)

    # ---- stage 3: rmsnorm + w3 ----
    ms = tl.sum(x2 * x2, axis=0) / D_size
    r = tl.math.rsqrt(ms + eps)
    xn = (x2 * r).to(tl.bfloat16).to(tl.float32)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0).to(tl.float32)
    out = (xn * w3).to(tl.bfloat16)

    tl.store(Y + base + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        if not (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 2):
            return self._forward_ref(x)

        x = x.contiguous()
        M_, D_ = x.shape
        y = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(D_)
        num_warps = 8 if BLOCK >= 2048 else 4
        _triple_rmsnorm_kernel[(M_,)](
            x, self.rms0_w, self.rms2_w, self.rms3_w, y,
            D_, 1.4041, 1e-6,
            BLOCK=BLOCK, num_warps=num_warps,
        )
        return y

    def _forward_ref(self, x):
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms0_w
        x = x * 1.4041
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
        _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms3_w
        return x
