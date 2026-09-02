import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 832
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _rms_kernel(X, W, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + offs).to(tl.float32)
    r = tl.math.rsqrt(tl.sum(x * x, 0) / N + 1e-6)
    xh = (x * r).to(tl.float16)
    w = tl.load(W + offs)  # fp16
    tl.store(Y + row * N + offs, xh * w)


@triton.jit
def _rms_scale_ln_kernel(X, W2, G, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X + row * N + offs).to(tl.float32)

    # RMSNorm (fp32 accumulation, fp16 cast, fp16 weight multiply)
    r = tl.math.rsqrt(tl.sum(x * x, 0) / N + 1e-6)
    y16 = (x * r).to(tl.float16) * tl.load(W2 + offs)

    # scalar multiply: PyTorch uses fp32 opmath then casts back to fp16
    y16 = (y16.to(tl.float32) * 1.1618).to(tl.float16)

    # LayerNorm (fp32 statistics, fp32 affine, fp16 output)
    yf = y16.to(tl.float32)
    mean = tl.sum(yf, 0) / N
    d = yf - mean
    var = tl.sum(d * d, 0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + offs).to(tl.float32)
    b = tl.load(B + offs).to(tl.float32)
    out = (d * rstd * g + b).to(tl.float16)
    tl.store(Y + row * N + offs, out)


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
        orig_shape = x.shape
        N = orig_shape[-1]
        x = x.contiguous().view(-1, N)
        rows = x.shape[0]
        BLOCK = triton.next_power_of_2(N)

        # Fused RMSNorm 0
        y0 = torch.empty_like(x)
        _rms_kernel[(rows,)](x, self.rms0_w, y0, N, BLOCK=BLOCK, num_warps=8)

        # Matmul via cuBLAS tensor cores
        h = y0 @ self.W1

        # Fused RMSNorm 2 + scale + LayerNorm 4
        out = torch.empty_like(h)
        _rms_scale_ln_kernel[(rows,)](
            h, self.rms2_w, self.ln4_g, self.ln4_b, out, N, BLOCK=BLOCK, num_warps=8
        )

        return out.view(orig_shape)
