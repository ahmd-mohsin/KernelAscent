import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 164
M, D, DT = 512, 2048, torch.bfloat16


@triton.jit
def _fused_norms_kernel(
    X, Y, W1, W2, G, B,
    stride_x, stride_y,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 1 (compute in fp32, round to bf16, mul by weight, round) ----
    ms1 = tl.sum(x * x, axis=0) / N
    y = (x * tl.math.rsqrt(ms1 + 1e-6)).to(tl.bfloat16)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (y.to(tl.float32) * w1).to(tl.bfloat16)

    # ---- RMSNorm 2 ----
    yf = y.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    z = (yf * tl.math.rsqrt(ms2 + 1e-6)).to(tl.bfloat16)
    w2 = tl.load(W2 + offs, mask=mask, other=0.0).to(tl.float32)
    z = (z.to(tl.float32) * w2).to(tl.bfloat16)

    # ---- scalar scale (fp32 opmath, round to bf16) ----
    z = (z.to(tl.float32) * 1.0772).to(tl.bfloat16)

    # ---- LayerNorm (fp32 accumulation) ----
    zf = z.to(tl.float32)
    zf = tl.where(mask, zf, 0.0)
    mean = tl.sum(zf, axis=0) / N
    diff = tl.where(mask, zf - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    out = (diff * rstd * g + b).to(tl.bfloat16)

    tl.store(Y + row * stride_y + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln4_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.W5 = nn.Parameter((torch.randn(4096, 512, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)

    def forward(self, x):
        if not x.is_cuda:
            # CPU fallback: reference path
            x = x @ self.W0
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = x * 1.0772
            x = F.layer_norm(x, (x.shape[-1],), self.ln4_g, self.ln4_b)
            return x @ self.W5

        x = x @ self.W0
        x = x.contiguous()
        rows, N = x.shape
        y = torch.empty_like(x)
        _fused_norms_kernel[(rows,)](
            x, y,
            self.rms1_w, self.rms2_w, self.ln4_g, self.ln4_b,
            x.stride(0), y.stride(0),
            N=N,
            BLOCK=triton.next_power_of_2(N),
            num_warps=16,
        )
        return y @ self.W5
