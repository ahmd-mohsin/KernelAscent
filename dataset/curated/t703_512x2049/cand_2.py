import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 703
M, D, DT = 512, 2049, torch.float16


@triton.jit
def _fused_norm_kernel(
    X, W1, G2, B2, W5, OUT,
    stride,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # ---- RMSNorm 1 (fp32 compute, round to fp16, then fp16*fp16 weight mul with fp32 opmath) ----
    ms = tl.sum(x * x, axis=0) / N
    x = (x * tl.math.rsqrt(ms + 1e-6)).to(tl.float16).to(tl.float32)
    w1 = tl.load(W1 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (x * w1).to(tl.float16).to(tl.float32)

    # ---- LayerNorm 2 (fp32 accumulation like PyTorch's half layer_norm) ----
    mean = tl.sum(x, axis=0) / N
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-5)
    g = tl.load(G2 + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B2 + offs, mask=mask, other=0.0).to(tl.float32)
    x = (xm * rstd * g + b).to(tl.float16).to(tl.float32)

    # ---- ReLU + scale (half*scalar uses fp32 opmath, rounds to fp16) ----
    x = tl.maximum(x, 0.0)
    x = (x * 1.455).to(tl.float16).to(tl.float32)

    # ---- RMSNorm 5 ----
    ms2 = tl.sum(x * x, axis=0) / N
    x = (x * tl.math.rsqrt(ms2 + 1e-6)).to(tl.float16).to(tl.float32)
    w5 = tl.load(W5 + offs, mask=mask, other=0.0).to(tl.float32)
    y = (x * w5).to(tl.float16)

    tl.store(OUT + row * stride + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2049, 512, generator=g) / math.sqrt(2049)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms5_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS tensor cores
        x = x @ self.W0
        x = x.contiguous()

        if not x.is_cuda:
            # CPU fallback: reference path
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln2_g, self.ln2_b)
            x = torch.relu(x)
            x = x * 1.455
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms5_w
            return x

        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norm_kernel[(Mrows,)](
            x, self.rms1_w, self.ln2_g, self.ln2_b, self.rms5_w, out,
            x.stride(0),
            N=N, BLOCK=BLOCK,
            num_warps=4,
        )
        return out
