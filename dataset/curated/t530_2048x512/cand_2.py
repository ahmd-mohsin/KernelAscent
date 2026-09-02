import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 530
M, D, DT = 2048, 512, torch.float16


@triton.jit
def _fused_softmax_rms_rms_ln(
    X, B1, W3, W4, G5, B5, OUT,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    base = row * N

    # bias add (fp16 semantics, matching x + b1 on half tensors)
    xh = tl.load(X + base + offs)                      # fp16
    bh = tl.load(B1 + offs)                            # fp16
    xh = xh + bh                                       # fp16 add

    # softmax in fp32 (PyTorch computes half softmax with float accumulation)
    xf = xh.to(tl.float32)
    mx = tl.max(xf, axis=0)
    ex = tl.exp(xf - mx)
    sm = tl.sum(ex, axis=0)
    p = ex / sm
    ph = p.to(tl.float16)                              # softmax output rounded to half

    # RMSNorm 1: xf = x.float(); (xf * rsqrt(mean(xf^2)+eps)).half() * w
    pf = ph.to(tl.float32)
    ms1 = tl.sum(pf * pf, axis=0) / N
    r1 = 1.0 / tl.sqrt(ms1 + 1e-6)
    w3 = tl.load(W3 + offs)                            # fp16
    y1 = (pf * r1).to(tl.float16) * w3                 # fp16 mul

    # RMSNorm 2
    yf = y1.to(tl.float32)
    ms2 = tl.sum(yf * yf, axis=0) / N
    r2 = 1.0 / tl.sqrt(ms2 + 1e-6)
    w4 = tl.load(W4 + offs)                            # fp16
    y2 = (yf * r2).to(tl.float16) * w4                 # fp16 mul

    # LayerNorm (fp32 internal, matching PyTorch half layer_norm)
    zf = y2.to(tl.float32)
    mean = tl.sum(zf, axis=0) / N
    d = zf - mean
    var = tl.sum(d * d, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(G5 + offs).to(tl.float32)
    b = tl.load(B5 + offs).to(tl.float32)
    out = d * rstd * g + b

    tl.store(OUT + base + offs, out.to(tl.float16))


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(512, 4096, generator=g) / math.sqrt(512)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln5_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        h = x @ self.W0                                # cuBLAS fp16 GEMM (tensor cores)
        h = h.contiguous()
        m, n = h.shape
        out = torch.empty_like(h)
        _fused_softmax_rms_rms_ln[(m,)](
            h, self.b1, self.rms3_w, self.rms4_w, self.ln5_g, self.ln5_b, out,
            N=n, BLOCK=n, num_warps=8,
        )
        return out
