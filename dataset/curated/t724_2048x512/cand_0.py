import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 724
M, D, DT = 2048, 512, torch.bfloat16


@triton.jit
def _gelu2_rms_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, stride,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2: tl.constexpr = 0.7071067811865476

    # first gelu (exact/erf), computed in fp32 then rounded to bf16 (matches PyTorch)
    g = 0.5 * x * (1.0 + tl.math.erf(x * INV_SQRT2))
    g = g.to(tl.bfloat16).to(tl.float32)

    # second gelu
    g = 0.5 * g * (1.0 + tl.math.erf(g * INV_SQRT2))
    g = g.to(tl.bfloat16)

    # RMS norm in fp32
    xf = g.to(tl.float32)
    xf = tl.where(mask, xf, 0.0)
    ms = tl.sum(xf * xf, axis=0) / N
    r = tl.math.rsqrt(ms + EPS)

    normed = (xf * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = (normed * w).to(tl.bfloat16)

    tl.store(Y_ptr + row * stride + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = x @ self.W0

        if not h.is_cuda:
            h = F.gelu(h)
            h = F.gelu(h)
            _xf = h.float()
            h = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(h.dtype) * self.rms3_w
            return h

        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        num_warps = 8 if BLOCK >= 2048 else 4
        _gelu2_rms_kernel[(Mrows,)](
            h, self.rms3_w, out,
            N, h.stride(0),
            EPS=1e-6,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out
