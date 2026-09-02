import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 832
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _rms_kernel(X, W, Y, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps)
    y = (x * r).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    tl.store(Y + row * N + cols, y * w, mask=mask)


@triton.jit
def _rms_scale_ln_kernel(X, W, G, B, Y, N, rms_eps, ln_eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    # RMSNorm (fp32 math, cast to fp16, weight mul in fp16 — matches reference)
    x = tl.load(X + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    ms = tl.sum(x * x, axis=0) / N
    r = 1.0 / tl.sqrt(ms + rms_eps)
    y = (x * r).to(tl.float16)
    w = tl.load(W + cols, mask=mask, other=0.0)
    z = y * w

    # scalar multiply in fp16 (matches: half_tensor * python_float)
    scale = tl.full([1], 1.1618, tl.float16)
    z = z * scale

    # LayerNorm computed in fp32 (matches F.layer_norm on fp16 inputs)
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    mean = tl.sum(zf, axis=0) / N
    d = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + ln_eps)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    out = d * rstd * g + b
    tl.store(Y + row * N + cols, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.rms0_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.W1 = nn.Parameter((torch.randn(2048, 2048, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x.contiguous()
        Mrows, N = x.shape
        BLOCK = triton.next_power_of_2(N)

        # Fused RMSNorm * rms0_w
        y = torch.empty_like(x)
        _rms_kernel[(Mrows,)](x, self.rms0_w, y, N, 1e-6, BLOCK=BLOCK, num_warps=8)

        # GEMM via cuBLAS tensor cores
        h = y @ self.W1

        # Fused RMSNorm * rms2_w -> *1.1618 -> LayerNorm
        out = torch.empty_like(h)
        _rms_scale_ln_kernel[(Mrows,)](
            h, self.rms2_w, self.ln4_g, self.ln4_b, out,
            N, 1e-6, 1e-5, BLOCK=BLOCK, num_warps=8,
        )
        return out
