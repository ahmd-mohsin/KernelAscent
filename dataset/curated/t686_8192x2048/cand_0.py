import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 686
M, D, DT = 8192, 2048, torch.float16


@triton.jit
def _fused_post_kernel(
    X, B1, LNG, LNB, W3, W4, B5, OUT,
    N,
    EPS_LN: tl.constexpr,
    EPS_RMS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # load matmul output row (fp16) and bias, add in fp16 (matches reference)
    x = tl.load(X + row * N + offs, mask=mask, other=0.0)
    b1 = tl.load(B1 + offs, mask=mask, other=0.0)
    x = x + b1  # fp16 + fp16 -> fp16

    # LayerNorm (computed in fp32 internally, like PyTorch on half input)
    xf = x.to(tl.float32)
    mean = tl.sum(xf, axis=0) / N
    xc = tl.where(mask, xf - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + EPS_LN)
    g = tl.load(LNG + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(LNB + offs, mask=mask, other=0.0).to(tl.float32)
    y = (xc * rstd * g + b).to(tl.float16)

    # RMSNorm 3: fp32 accum, cast to fp16, multiply by weight in fp16
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    w3 = tl.load(W3 + offs, mask=mask, other=0.0)
    y = (yf * r).to(tl.float16) * w3

    # RMSNorm 4
    yf = y.to(tl.float32)
    ms = tl.sum(yf * yf, axis=0) / N
    r = 1.0 / tl.sqrt(ms + EPS_RMS)
    w4 = tl.load(W4 + offs, mask=mask, other=0.0)
    y = (yf * r).to(tl.float16) * w4

    # bias add in fp16
    b5 = tl.load(B5 + offs, mask=mask, other=0.0)
    y = y + b5

    tl.store(OUT + row * N + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(2048, 4096, generator=g) / math.sqrt(2048)).to(dtype), requires_grad=False)
        self.b1 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_g = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.ln2_b = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms3_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)
        self.b5 = nn.Parameter(torch.randn(4096, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM via cuBLAS (tensor cores on A100)
        h = torch.matmul(x, self.W0)
        if not h.is_contiguous():
            h = h.contiguous()

        Mrows, N = h.shape
        out = torch.empty_like(h)
        BLOCK = triton.next_power_of_2(N)
        _fused_post_kernel[(Mrows,)](
            h, self.b1, self.ln2_g, self.ln2_b,
            self.rms3_w, self.rms4_w, self.b5, out,
            N,
            EPS_LN=1e-5,
            EPS_RMS=1e-6,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out
