import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 633
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_norms_gelu_kernel(
    X, OUT, W1, W2, G, B,
    N, stride_x, stride_o,
    EPS_RMS: tl.constexpr, EPS_LN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N

    x16 = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    x = x16.to(tl.float32)

    # RMSNorm 1 (fp32 math, cast to fp16, fp16 multiply by weight)
    ms1 = tl.sum(x * x, axis=0) / N
    y = (x * tl.math.rsqrt(ms1 + EPS_RMS)).to(tl.float16)
    w1 = tl.load(W1 + cols, mask=mask, other=0.0)
    y = y * w1  # fp16 arithmetic

    # RMSNorm 2
    x = y.to(tl.float32)
    ms2 = tl.sum(x * x, axis=0) / N
    y = (x * tl.math.rsqrt(ms2 + EPS_RMS)).to(tl.float16)
    w2 = tl.load(W2 + cols, mask=mask, other=0.0)
    y = y * w2  # fp16 arithmetic

    # LayerNorm (fp32 accumulation)
    x = y.to(tl.float32)
    mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    inv = tl.math.rsqrt(var + EPS_LN)
    g = tl.load(G + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
    y = ((x - mean) * inv * g + b).to(tl.float16)

    # GELU (exact, fp32 math like PyTorch opmath)
    xf = y.to(tl.float32)
    out = (xf * 0.5 * (1.0 + tl.math.erf(xf * 0.7071067811865476))).to(tl.float16)

    tl.store(OUT + row * stride_o + cols, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 512, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.rms1_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        x = x @ self.W0
        if not x.is_cuda:
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms1_w
            _xf = x.float(); x = (_xf * torch.rsqrt(_xf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.rms2_w
            x = F.layer_norm(x, (x.shape[-1],), self.ln3_g, self.ln3_b)
            return F.gelu(x)

        x = x.contiguous()
        Mrows, N = x.shape
        out = torch.empty_like(x)
        BLOCK = triton.next_power_of_2(N)
        _fused_norms_gelu_kernel[(Mrows,)](
            x, out, self.rms1_w, self.rms2_w, self.ln3_g, self.ln3_b,
            N, x.stride(0), out.stride(0),
            EPS_RMS=1e-6, EPS_LN=1e-5,
            BLOCK=BLOCK, num_warps=4,
        )
        return out
