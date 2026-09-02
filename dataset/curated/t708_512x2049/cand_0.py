import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 708
M, D, DT = 512, 2049, torch.bfloat16

_INV_SQRT2 = 0.7071067811865476


@triton.jit
def _gelu_rms_kernel(X, W, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    # exact GELU (erf), computed in fp32 then rounded to bf16 (matches PyTorch opmath)
    g = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    g = g.to(tl.bfloat16).to(tl.float32)
    # RMSNorm in fp32
    ms = tl.sum(g * g, axis=0) / N
    r = tl.math.rsqrt(ms + 1e-6)
    h = (g * r).to(tl.bfloat16).to(tl.float32)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    out = (h * w).to(tl.bfloat16)
    tl.store(Y + row * N + offs, out, mask=mask)


@triton.jit
def _scale_gelu_kernel(X, Y, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    t = (x * 1.3239).to(tl.bfloat16).to(tl.float32)
    g = 0.5 * t * (1.0 + tl.math.erf(t * 0.7071067811865476))
    tl.store(Y + offs, g.to(tl.bfloat16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 2048, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W3 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # fallback: reference path
            x = x @ self.W0
            x = F.gelu(x)
            _xf = x.float()
            x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = x @ self.W3
            x = x * 1.3239
            x = F.gelu(x)
            return x

        # matmul 1 (cuBLAS)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N = h.shape

        # fused GELU + RMSNorm + weight
        out1 = torch.empty_like(h)
        _gelu_rms_kernel[(Mrows,)](
            h, self.rms2_w, out1,
            N=N, BLOCK=triton.next_power_of_2(N),
            num_warps=8,
        )

        # matmul 2 (cuBLAS)
        y = torch.matmul(out1, self.W3)
        y = y.contiguous()

        # fused scale + GELU (in-place on y buffer)
        n = y.numel()
        BLOCK = 1024
        _scale_gelu_kernel[(triton.cdiv(n, BLOCK),)](
            y, y, n, BLOCK=BLOCK, num_warps=4,
        )
        return y
