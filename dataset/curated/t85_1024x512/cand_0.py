import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 85
M, D, DT = 1024, 512, torch.float16


@triton.jit
def _ln_rms_scale_kernel(
    X, G, B, W, Y,
    N, stride,
    eps_ln, eps_rms, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * stride + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 math, matching PyTorch's half->float acc path)
    mean = tl.sum(x, axis=0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps_ln)

    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = d * rstd * g + b
    yh = y.to(tl.float16)          # LN output rounded to fp16 (as in reference)

    # RMSNorm: cast fp16 -> fp32, normalize, cast back to fp16
    yf = yh.to(tl.float32)
    ms = tl.sum(tl.where(mask, yf * yf, 0.0), axis=0) / N
    r = 1.0 / tl.sqrt(ms + eps_rms)
    t = (yf * r).to(tl.float16)    # matches .to(x.dtype)

    # * rms2_w  (half*half elementwise: fp32 opmath, fp16 round)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    u = (t.to(tl.float32) * w).to(tl.float16)

    # * 1.3885  (fp32 opmath, fp16 round)
    out = (u.to(tl.float32) * scale).to(tl.float16)

    tl.store(Y + row * stride + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 2048, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.ln1_g = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.ln1_b = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)
        self.rms2_w = nn.Parameter(torch.randn(2048, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores)
        h = x @ self.W0
        h = h.contiguous()
        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _ln_rms_scale_kernel[(Mrows,)](
            h, self.ln1_g, self.ln1_b, self.rms2_w, out,
            N, h.stride(0),
            1e-5, 1e-6, 1.3885,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
