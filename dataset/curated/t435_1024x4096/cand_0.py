import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

SEED = 435
M, D, DT = 1024, 4096, torch.float16


@triton.jit
def _softmax_kernel(X, Y, N, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + row * N + offs, mask=mask, other=-float('inf')).to(tl.float32)
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    y = e / s
    tl.store(Y + row * N + offs, y.to(tl.float16), mask=mask)


@triton.jit
def _ln_rms_gelu_kernel(X, G, B, W, Y, N, eps_ln, eps_rms, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X + row * N + offs, mask=mask, other=0.0).to(tl.float32)

    # LayerNorm (fp32 internal math, like F.layer_norm on fp16)
    mean = tl.sum(x, 0) / N
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / N
    inv = 1.0 / tl.sqrt(var + eps_ln)
    g = tl.load(G + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(B + offs, mask=mask, other=0.0).to(tl.float32)
    y = (d * inv) * g + b
    # layer_norm output is fp16 in the reference; round here to match
    y16 = y.to(tl.float16)

    # RMSNorm step: xf = y16.float(); (xf * rsqrt(mean(xf^2)+eps)).half() * w
    yf = y16.to(tl.float32)
    yf = tl.where(mask, yf, 0.0)
    rms = 1.0 / tl.sqrt(tl.sum(yf * yf, 0) / N + eps_rms)
    w = tl.load(W + offs, mask=mask, other=0.0)  # fp16
    z16 = (yf * rms).to(tl.float16) * w          # fp16 multiply as in reference
    zf = z16.to(tl.float32)

    # GELU (erf variant, fp32 internal math like CUDA half gelu)
    out = zf * 0.5 * (1.0 + tl.math.erf(zf * 0.7071067811865476))
    tl.store(Y + row * N + offs, out.to(tl.float16), mask=mask)


class ModelNew(nn.Module):
    def __init__(self, dtype=DT):
        super().__init__()
        g = torch.Generator().manual_seed(SEED)
        self.W0 = nn.Parameter((torch.randn(4096, 1024, generator=g) / math.sqrt(4096)).to(dtype), requires_grad=False)
        self.W2 = nn.Parameter((torch.randn(1024, 512, generator=g) / math.sqrt(1024)).to(dtype), requires_grad=False)
        self.ln3_g = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.ln3_b = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)
        self.rms4_w = nn.Parameter(torch.randn(512, generator=g).to(dtype), requires_grad=False)

    def forward(self, x):
        # GEMM 1 (cuBLAS tensor cores)
        h = torch.matmul(x, self.W0)
        h = h.contiguous()
        Mrows, N1 = h.shape

        # Fused row softmax (fp32 accumulation)
        s = torch.empty_like(h)
        _softmax_kernel[(Mrows,)](
            h, s, N1,
            BLOCK=triton.next_power_of_2(N1),
            num_warps=8,
        )

        # GEMM 2 (cuBLAS tensor cores)
        y = torch.matmul(s, self.W2).contiguous()
        Mr2, N2 = y.shape

        # Fused LayerNorm + RMSNorm + GELU
        out = torch.empty_like(y)
        _ln_rms_gelu_kernel[(Mr2,)](
            y, self.ln3_g, self.ln3_b, self.rms4_w, out,
            N2, 1e-5, 1e-6,
            BLOCK=triton.next_power_of_2(N2),
            num_warps=4,
        )
        return out
