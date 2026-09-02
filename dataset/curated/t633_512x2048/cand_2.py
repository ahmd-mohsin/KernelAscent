import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 633
M, D, DT = 512, 2048, torch.float16


@triton.jit
def _fused_norms_gelu(X, W1, W2, G, B, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    base = row * N

    x = tl.load(X + base + cols).to(tl.float32)

    # RMSNorm 1 (matching reference rounding: fp32 norm -> fp16 -> fp16*fp16 mul in fp32 -> fp16)
    rs1 = tl.rsqrt(tl.sum(x * x, axis=0) / N + 1e-6)
    t = (x * rs1).to(tl.float16)
    w1 = tl.load(W1 + cols).to(tl.float32)
    t = (t.to(tl.float32) * w1).to(tl.float16)

    # RMSNorm 2
    x2 = t.to(tl.float32)
    rs2 = tl.rsqrt(tl.sum(x2 * x2, axis=0) / N + 1e-6)
    t2 = (x2 * rs2).to(tl.float16)
    w2 = tl.load(W2 + cols).to(tl.float32)
    t2 = (t2.to(tl.float32) * w2).to(tl.float16)

    # LayerNorm (fp32 internal math, eps=1e-5, output rounded to fp16)
    x3 = t2.to(tl.float32)
    mean = tl.sum(x3, axis=0) / N
    xc = x3 - mean
    var = tl.sum(xc * xc, axis=0) / N
    rstd = tl.rsqrt(var + 1e-5)
    g = tl.load(G + cols).to(tl.float32)
    b = tl.load(B + cols).to(tl.float32)
    y16 = (xc * rstd * g + b).to(tl.float16)

    # Exact GELU (erf), computed in fp32 on the fp16-rounded layernorm output
    yf = y16.to(tl.float32)
    out = 0.5 * yf * (1.0 + tl.math.erf(yf * 0.7071067811865476))

    tl.store(Y + base + cols, out.to(tl.float16))


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
        x = x @ self.W0  # cuBLAS half GEMM (tensor cores)
        x = x.contiguous()
        Mrows, N = x.shape
        y = torch.empty_like(x)
        _fused_norms_gelu[(Mrows,)](
            x, self.rms1_w, self.rms2_w, self.ln3_g, self.ln3_b, y,
            N=N, BLOCK=N, num_warps=4,
        )
        return y
